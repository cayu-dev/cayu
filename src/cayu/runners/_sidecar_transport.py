"""HTTP transport and errors shared by runners that drive the Cayu command sidecar.

The sidecar speaks one protocol wherever it runs. A runner supplies how its
host authenticates and addresses it; the request handling, response bounds and
error classification here are common.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import httpx

from cayu.providers._http import SharedAsyncClient

SIDECAR_PROTOCOL_VERSION = "8"
# Sized for the built-in browser worker: a default session response envelope
# plus a browser-profile checkpoint, and a default upload batch on stdin. Both
# match the first-party sidecar's own ceilings.
SIDECAR_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
SIDECAR_MAX_STDIN_BYTES = 24 * 1024 * 1024
SIDECAR_MAX_ENCODED_OUTPUT_BYTES = 4 * ((SIDECAR_MAX_OUTPUT_BYTES + 2) // 3)
SIDECAR_MAX_RESPONSE_BYTES = 2 * SIDECAR_MAX_ENCODED_OUTPUT_BYTES + 64 * 1024
# 429 is handled on its own, as a throttle.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 500, 502, 503, 504})
_OWNER_SUPERSEDED_HTTP_STATUS = 412
_STALE_OWNER_EPOCH_HTTP_STATUS = 409
_OWNER_LIFECYCLE_LEASED_HTTP_STATUS = 423


class SidecarRunnerError(RuntimeError):
    """Base error for runners that drive the Cayu command sidecar."""


class SidecarProtocolError(SidecarRunnerError):
    """A sidecar response violated the runner contract."""


class _SidecarProtocolVersionMismatch(SidecarProtocolError):
    """The sidecar is healthy but speaks an incompatible protocol version."""


class SidecarEndpointUnauthorized(SidecarRunnerError):
    """The sidecar rejected the host's credential."""


class SidecarEndpointTransientError(SidecarRunnerError):
    """A transport failure eligible for bounded command-state read retry only."""


class SidecarEndpointThrottled(SidecarEndpointTransientError):
    """The endpoint answered 429: it refused the request before acting on it.

    Any endpoint call may be sent again after a pause; ``retry_after_s`` is the
    endpoint's ``Retry-After``, when it sends one."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class SidecarOwnershipSuperseded(SidecarRunnerError):
    """A later host claimed the sidecar; this runner may no longer act on it.

    Execution is permanently closed so a stale owner cannot run commands in its
    successor's workspace.
    """


class SidecarStaleOwnerEpoch(SidecarOwnershipSuperseded):
    """The requested owner epoch is older than one the sidecar already accepted.

    The current owner and its work are unchanged. Retrying the same stale epoch
    cannot acquire ownership; the rejected runner must not act on the sidecar.
    """


class SidecarOwnershipUnverified(SidecarRunnerError):
    """The sidecar could not confirm this runner still owns it."""


class SidecarLifecycleInProgress(SidecarRunnerError):
    """The current owner is suspending or terminating the sandbox."""


class _SidecarAdmissionIdentityDrift(SidecarRunnerError):
    """The exact sandbox changed while admission evidence was being observed."""


@dataclass(frozen=True, slots=True)
class SidecarErrors:
    """The error classes and message label one runner reports sidecar failures with."""

    label: str
    error: type[SidecarRunnerError]
    protocol: type[SidecarProtocolError]
    version_mismatch: type[_SidecarProtocolVersionMismatch]
    unauthorized: type[SidecarEndpointUnauthorized]
    transient: type[SidecarEndpointTransientError]
    throttled: type[SidecarEndpointThrottled]
    superseded: type[SidecarOwnershipSuperseded]
    stale_owner_epoch: type[SidecarStaleOwnerEpoch]
    lifecycle_in_progress: type[SidecarLifecycleInProgress]
    identity_drift: type[_SidecarAdmissionIdentityDrift]


SIDECAR_ERRORS = SidecarErrors(
    label="Sidecar",
    error=SidecarRunnerError,
    protocol=SidecarProtocolError,
    version_mismatch=_SidecarProtocolVersionMismatch,
    unauthorized=SidecarEndpointUnauthorized,
    transient=SidecarEndpointTransientError,
    throttled=SidecarEndpointThrottled,
    superseded=SidecarOwnershipSuperseded,
    stale_owner_epoch=SidecarStaleOwnerEpoch,
    lifecycle_in_progress=SidecarLifecycleInProgress,
    identity_drift=_SidecarAdmissionIdentityDrift,
)


class SidecarEndpointTransport(Protocol):
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> Mapping[str, Any]: ...

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
        owner_claim: str | None = None,
    ) -> Mapping[str, Any]: ...

    async def claim_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
        owner_epoch: int | None = None,
        carry_commands: Sequence[str] | None = None,
    ) -> Mapping[str, Any]: ...

    async def check_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...


class HttpxSidecarEndpointTransport:
    """HTTP transport for the Cayu command sidecar.

    Subclasses say how the host authenticates (``_auth_headers``) and where the
    sidecar listens (``_base_url``).
    """

    _errors: ClassVar[SidecarErrors] = SIDECAR_ERRORS

    def __init__(self) -> None:
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _auth_headers(self, token: str) -> dict[str, str]:
        raise NotImplementedError

    def _base_url(self, endpoint: str) -> str:
        raise NotImplementedError

    def _max_response_bytes(self) -> int:
        return SIDECAR_MAX_RESPONSE_BYTES

    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> Mapping[str, Any]:
        response = await self._request(
            "GET", endpoint=endpoint, token=token, path="/health", timeout_s=timeout_s
        )
        return response

    async def start_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/commands",
            payload={"command_id": command_id, **payload},
            timeout_s=timeout_s,
        )

    async def get_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "GET",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}",
            timeout_s=timeout_s,
        )

    async def cancel_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
        owner_claim: str | None = None,
    ) -> Mapping[str, Any]:
        return await self._request(
            "DELETE",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}",
            timeout_s=timeout_s,
            payload=None if owner_claim is None else {"owner_claim": owner_claim},
        )

    async def release_command(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        timeout_s: float,
        owner_claim: str | None = None,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}/release",
            timeout_s=timeout_s,
            payload=None if owner_claim is None else {"owner_claim": owner_claim},
        )

    async def write_command_stdin(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        owner_claim: str,
        offset: int,
        data_base64: str,
        eof: bool,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}/stdin",
            payload={
                "owner_claim": owner_claim,
                "offset": offset,
                "data_base64": data_base64,
                "eof": eof,
            },
            timeout_s=timeout_s,
        )

    async def read_command_stdout(
        self,
        *,
        endpoint: str,
        token: str,
        command_id: str,
        owner_claim: str,
        offset: int,
        max_bytes: int,
        wait_s: float,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/commands/{command_id}/stdout",
            payload={
                "owner_claim": owner_claim,
                "offset": offset,
                "max_bytes": max_bytes,
                "wait_s": wait_s,
            },
            timeout_s=timeout_s,
        )

    async def open_forward(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        owner_claim: str,
        lifetime_s: int | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/forwards",
            payload={"forward_id": forward_id, "owner_claim": owner_claim, "timeout_s": lifetime_s},
            timeout_s=timeout_s,
        )

    async def accept_forward(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        owner_claim: str,
        after: int,
        wait_s: float,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/forwards/{forward_id}/accept",
            payload={"owner_claim": owner_claim, "after": after, "wait_s": wait_s},
            timeout_s=timeout_s,
        )

    async def read_forward(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        connection: int,
        owner_claim: str,
        offset: int,
        max_bytes: int,
        wait_s: float,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/forwards/{forward_id}/connections/{connection}/read",
            payload={
                "owner_claim": owner_claim,
                "offset": offset,
                "max_bytes": max_bytes,
                "wait_s": wait_s,
            },
            timeout_s=timeout_s,
        )

    async def write_forward(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        connection: int,
        owner_claim: str,
        offset: int,
        data_base64: str,
        eof: bool,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/forwards/{forward_id}/connections/{connection}/write",
            payload={
                "owner_claim": owner_claim,
                "offset": offset,
                "data_base64": data_base64,
                "eof": eof,
            },
            timeout_s=timeout_s,
        )

    async def close_forward_connection(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        connection: int,
        owner_claim: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/forwards/{forward_id}/connections/{connection}/close",
            payload={"owner_claim": owner_claim},
            timeout_s=timeout_s,
        )

    async def close_forward(
        self,
        *,
        endpoint: str,
        token: str,
        forward_id: str,
        owner_claim: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path=f"/v1/forwards/{forward_id}/close",
            payload={"owner_claim": owner_claim},
            timeout_s=timeout_s,
        )

    async def claim_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
        owner_epoch: int | None = None,
        carry_commands: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner",
            payload={
                "claim_id": claim_id,
                **({"owner_epoch": owner_epoch} if owner_epoch is not None else {}),
                **({"carry_commands": list(carry_commands)} if carry_commands is not None else {}),
            },
            timeout_s=timeout_s,
        )

    async def check_owner(
        self,
        *,
        endpoint: str,
        token: str,
        claim_id: str,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._request(
            "POST",
            endpoint=endpoint,
            token=token,
            path="/v1/owner/check",
            payload={"claim_id": claim_id},
            timeout_s=timeout_s,
        )

    async def _request(
        self,
        method: str,
        *,
        endpoint: str,
        token: str,
        path: str,
        timeout_s: float,
        payload: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        errors = self._errors
        label = errors.label
        max_response_bytes = self._max_response_bytes()
        headers = {**self._auth_headers(token), "Accept-Encoding": "identity"}
        try:
            request_options: dict[str, Any] = {"headers": headers, "timeout": timeout_s}
            if payload is not None:
                request_options["json"] = payload
            async with asyncio.timeout(timeout_s):
                async with self._client.get().stream(
                    method, f"{self._base_url(endpoint)}{path}", **request_options
                ) as response:
                    if response.status_code in {401, 403}:
                        raise errors.unauthorized(f"{label} endpoint token was rejected.")
                    if (
                        path == "/v1/owner"
                        and response.status_code == _STALE_OWNER_EPOCH_HTTP_STATUS
                    ):
                        raise errors.stale_owner_epoch(
                            f"{label} sidecar rejected an older owner epoch."
                        )
                    if response.status_code == _OWNER_SUPERSEDED_HTTP_STATUS:
                        raise errors.superseded(f"{label} sidecar reports a newer owner.")
                    if response.status_code == _OWNER_LIFECYCLE_LEASED_HTTP_STATUS:
                        raise errors.lifecycle_in_progress(
                            f"{label} owner is suspending or terminating it."
                        )
                    if response.status_code == 429:
                        raise errors.throttled(
                            f"{label} endpoint returned HTTP 429; response body omitted",
                            retry_after_s=_retry_after_seconds(response.headers.get("retry-after")),
                        )
                    if response.status_code >= 400:
                        error_type = (
                            errors.transient
                            if response.status_code in _TRANSIENT_HTTP_STATUSES
                            else errors.error
                        )
                        raise error_type(
                            f"{label} endpoint returned HTTP {response.status_code}; "
                            "response body omitted"
                        )
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise errors.protocol(
                            f"{label} endpoint returned unsupported content encoding."
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes(
                        chunk_size=min(64 * 1024, max_response_bytes + 1)
                    ):
                        if len(chunk) > max_response_bytes - len(body):
                            raise errors.protocol(
                                f"{label} endpoint response exceeds its byte ceiling."
                            )
                        body.extend(chunk)
        except (httpx.TimeoutException, httpx.NetworkError, TimeoutError) as exc:
            raise errors.transient(f"{label} endpoint request temporarily failed.") from exc
        except httpx.RequestError as exc:
            raise errors.error(f"{label} endpoint request failed.") from exc
        try:
            decoded = json.loads(body)
        except ValueError:
            decoded = None
        finally:
            body.clear()
        if decoded is None:
            raise errors.protocol(f"{label} endpoint returned invalid JSON.")
        if not isinstance(decoded, Mapping):
            raise errors.protocol(f"{label} endpoint response must be an object.")
        return decoded


def _retry_after_seconds(value: str | None) -> float | None:
    """``Retry-After`` in seconds, when it is a whole number of them; dates aren't used."""

    if value is None or not (value := value.strip()).isdigit():
        return None
    return float(value)
