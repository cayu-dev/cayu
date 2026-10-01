"""Explicit opt-in transport for the implemented workload-policy contract.

Enrollment and credential provisioning remain controller-owned. This channel
cannot create an incarnation, renew authority, or enable inference credentials.
"""

from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import suppress
from typing import Any
from urllib.parse import quote, urlsplit

from cayu.providers._credential_boundary import credential_safe_provider_cancellation
from cayu.providers._http import SharedAsyncClient, _read_bounded_identity_error_response
from cayu.runtime._model_policy import PolicyChannel
from cayu.runtime._policy_contract import _decision, _integer, _report_receipt, _scope, _snapshot
from cayu.runtime._policy_wire import canonical, decode, identifier, require

_ERROR_STATUS = {
    "invalid_request": 422,
    "unauthenticated": 401,
    "forbidden": 403,
    "instance_retired": 409,
    "incarnation_conflict": 409,
    "operation_conflict": 409,
    "integrity_conflict": 409,
    "snapshot_unknown": 409,
    "snapshot_expired": 409,
    "adoption_unavailable": 409,
    "rate_limited": 429,
    "unavailable": 503,
}


class PolicyTransportUnavailable(RuntimeError):
    """No authenticated response; pending reports retain their exact identity."""


class PolicyResponseError(PolicyTransportUnavailable):
    """Finite authenticated refusal; never evidence that a report did not commit."""

    def __init__(self, code: str) -> None:
        require(type(code) is str and code in _ERROR_STATUS)
        self.code = code
        self.retryable = code in {"rate_limited", "unavailable"}
        super().__init__("Policy service rejected the operation: " + code + ".")


def _error(value: dict, status: int) -> str:
    require(set(value) == {"schema_version", "kind", "request_id", "code", "retryable"})
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "policy_error")
    identifier(value["request_id"])
    code = value["code"]
    require(type(code) is str and code in _ERROR_STATUS and _ERROR_STATUS[code] == status)
    require(type(value["retryable"]) is bool)
    require(value["retryable"] == (code in {"rate_limited", "unavailable"}))
    return code


def _bounded_document(wire: bytes) -> dict:
    value = decode(wire, max_bytes=65536)
    pending: list[tuple[Any, int]] = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        require(count <= 4096 and depth <= 12 and type(item) is not float)
        if type(item) is int:
            require(item >= 0)
        if type(item) is dict:
            pending.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif type(item) is list:
            pending.extend((part, depth + 1) for part in item)
    return value


class HttpPolicyChannel(PolicyChannel):
    """Explicit management origin, frozen workload credential and enrollment.

    No redirects, automatic retries, ambient inference key, or credential
    rotation. The controller/journal validate snapshots and exact receipts before
    installation or acknowledgement; a successful transport is not adoption.
    """

    def __init__(
        self,
        *,
        origin: str,
        scope: dict[str, str],
        incarnation: tuple[str, int],
        credential_env: str,
    ) -> None:
        require(type(origin) is str)
        parsed = None
        with suppress(ValueError):
            parsed = urlsplit(origin)
        require(parsed is not None)
        assert parsed is not None
        require(
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and "?" not in origin
            and "#" not in origin
            and all(33 <= ord(char) <= 126 for char in origin)
        )
        _scope(scope)
        require(type(incarnation) is tuple and len(incarnation) == 2)
        identifier(incarnation[0])
        _integer(incarnation[1])
        require(
            type(credential_env) is str
            and credential_env.isidentifier()
            and credential_env != "CAYU_GATEWAY_API_KEY"
        )
        key = os.environ.get(credential_env, "")
        require(0 < len(key) <= 4096 and all(33 <= ord(char) <= 126 for char in key))
        require(key not in origin and key.encode() not in canonical(scope))
        require(key not in incarnation[0])
        instance = scope["instance_id"]
        require(instance not in {".", ".."})
        self._url = origin.rstrip("/") + "/v1/model-policy/instances/" + quote(instance, safe="")
        self._scope = dict(scope)
        self._incarnation = incarnation
        self._credential_env = credential_env
        self._key = key
        self._client = SharedAsyncClient()

    @property
    def scope(self) -> dict[str, str]:
        return dict(self._scope)

    @property
    def incarnation(self) -> tuple[str, int]:
        return self._incarnation

    async def read_snapshot(self) -> bytes:
        wire = await self._request("GET", "/snapshot")
        value = _snapshot(wire, scope=self._scope)
        require((value["incarnation_id"], value["incarnation_epoch"]) == self._incarnation)
        return wire

    def _expected_report(self, report: bytes) -> bytes:
        value = _decision(report)
        require(value["scope"] == self._scope)
        require((value["incarnation_id"], value["incarnation_epoch"]) == self._incarnation)
        _bounded_document(report)
        body = canonical(value)
        require(self._key.encode() not in body)
        return body

    async def report(self, report: bytes) -> bytes:
        body = self._expected_report(report)
        path = "/refusals" if _decision(body)["kind"] == "adoption_refusal" else "/reports"
        wire = await self._request("POST", path, body=body)
        _report_receipt(wire, expected_report=body)
        return wire

    async def read_report(self, *, expected_report: bytes) -> bytes:
        """Exact read-only reconciliation; absence never settles responsibility."""
        body = self._expected_report(expected_report)
        operation = _decision(body)["operation_id"]
        require(operation not in {".", ".."})
        path = "/refusals/" if _decision(body)["kind"] == "adoption_refusal" else "/reports/"
        wire = await self._request("GET", path + quote(operation, safe=""))
        _report_receipt(wire, expected_report=body)
        return wire

    async def _request(self, method: str, path: str, *, body: bytes | None = None) -> bytes:
        require(
            hmac.compare_digest(
                os.environ.get(self._credential_env, "").encode(), self._key.encode()
            )
        )
        require(body is None or self._key.encode() not in body)
        result = None
        error = None
        cancellation = None
        task = asyncio.current_task()
        cancellation_baseline = task.cancelling() if task is not None else 0
        try:
            async with asyncio.timeout(10):
                async with self._client.get().stream(
                    method,
                    self._url + path,
                    content=body,
                    headers={
                        "Authorization": "Bearer " + self._key,
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                        "Content-Type": "application/json",
                    },
                    timeout=10,
                    follow_redirects=False,
                ) as response:
                    require(
                        response.status_code == 200
                        or response.status_code in _ERROR_STATUS.values()
                    )
                    require(response.headers.get("content-encoding", "identity") == "identity")
                    require(
                        response.headers.get("content-type", "").split(";")[0].strip()
                        == "application/json"
                    )
                    bounded = await _read_bounded_identity_error_response(
                        response, idle_timeout_s=10, max_duration_s=10
                    )
                    require(bounded is not None)
                    assert bounded is not None
                    require(self._key.encode() not in bounded.content)
                    document = _bounded_document(bounded.content)
                    result = canonical(document, max_bytes=65536)
                    require(self._key.encode() not in result)
                    if response.status_code != 200:
                        error = _error(document, response.status_code)
        except asyncio.CancelledError:
            if task is not None and task.cancelling() > cancellation_baseline:
                cancellation = credential_safe_provider_cancellation(
                    "Policy request cancelled.", preserve_empty_artifacts=True
                )
            else:
                result = None
        except Exception:
            result = None
            error = None
        if task is not None and task.cancelling() > cancellation_baseline and cancellation is None:
            cancellation = credential_safe_provider_cancellation(
                "Policy request cancelled.", preserve_empty_artifacts=True
            )
        if cancellation is not None:
            raise cancellation
        if result is None:
            raise PolicyTransportUnavailable("Authenticated policy response is unavailable.")
        if error is not None:
            raise PolicyResponseError(error)
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
