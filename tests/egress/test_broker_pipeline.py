from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import pytest
from pydantic import SecretStr

import cayu.egress._resolution as resolution_module
import cayu.egress.broker as broker_module
from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    CapturedRequest,
    CapturedResponse,
    EgressDecision,
    EgressPolicy,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.broker import CAYU_EGRESS_ERROR_HEADER
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault

REAL_SECRET = "sk_test_51RealDeadBeefSecretValue"
GITHUB_SECRET = "github_pat_11RealDeadBeefSecretValue"


def _destination_resolver(*addresses: str):  # type: ignore[no-untyped-def]
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return addresses

    return resolve


class _RecordingResolver:
    """Counts resolutions so tests can prove deny-before-resolve."""

    def __init__(self, vault: StaticVault) -> None:
        self._vault = vault
        self.resolve_count = 0

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        self.resolve_count += 1
        return await self._vault.resolve(ref, scope=scope)


class _FakeUpstream:
    def __init__(self, response: CapturedResponse) -> None:
        self._response = response
        self.sent: CapturedRequest | None = None

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            self.sent = request
            return self._response

        return EgressUpstreamOperation(send)


class _FailingUpstream:
    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            raise RuntimeError(f"boom with {REAL_SECRET}")  # secret in exception must not leak out

        return EgressUpstreamOperation(send)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _build(
    *,
    upstream: Any,
    clock: _Clock | None = None,
    policies: Mapping[str, EgressPolicy] | None = None,
    max_active_upstream_operations: int = 16,
) -> tuple[
    TransparentEgressBroker, VirtualCredentialRegistry, _RecordingResolver, list[EgressDecision]
]:
    registry = VirtualCredentialRegistry(clock=clock or _Clock(datetime(2026, 7, 6, tzinfo=UTC)))
    resolver = _RecordingResolver(
        StaticVault(
            {
                "github_token": GITHUB_SECRET,
                "stripe_test_key": REAL_SECRET,
            }
        )
    )
    decisions: list[EgressDecision] = []
    if policies is None:
        policies = {"stripe-example": _stripe_example_policy()}
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=resolver,
        policies=policies,
        upstream=upstream,
        audit=decisions.append,
        max_active_upstream_operations=max_active_upstream_operations,
    )
    return broker, registry, resolver, decisions


def _mint(registry: VirtualCredentialRegistry, **overrides: Any):
    params: dict[str, Any] = {
        "session_id": "sess_1",
        "env_name": "STRIPE_SECRET_KEY",
        "secret": SecretRef(name="stripe_test_key"),
        "destination": "api.stripe.com",
        "credential_kind": "stripe_bearer",
        "policy_name": "stripe-example",
    }
    params.update(overrides)
    return registry.mint(**params)


def _stripe_example_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name="stripe-example",
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
        denied_prefixes=["/v1/payouts"],
    )


def test_connect_destination_requires_current_positive_authority() -> None:
    clock = _Clock(datetime(2026, 7, 6, tzinfo=UTC))
    broker, registry, _resolver, _decisions = _build(
        upstream=_FakeUpstream(CapturedResponse(status_code=200)),
        clock=clock,
    )
    grant = _mint(registry, ttl_seconds=1)

    assert asyncio.run(broker.authorize_connect_destination(host="api.stripe.com", port=443))
    clock.advance(2)
    assert not asyncio.run(broker.authorize_connect_destination(host="api.stripe.com", port=443))

    replacement = _mint(registry)
    assert asyncio.run(broker.authorize_connect_destination(host="api.stripe.com", port=443))
    registry.revoke(replacement.presented_value)
    assert not asyncio.run(broker.authorize_connect_destination(host="api.stripe.com", port=443))
    assert grant.grant_id != replacement.grant_id


def test_connect_destination_accepts_only_active_credentialless_authority() -> None:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=None,
        policies={
            "browser": BrowserEgressPolicy(
                name="browser",
                allowed_hosts=["docs.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="browser",
            )
        ],
        upstream=_FakeUpstream(CapturedResponse(status_code=200)),
    )

    assert asyncio.run(broker.authorize_connect_destination(host="docs.example.com", port=443))
    asyncio.run(broker.revoke_authority_and_wait(()))
    assert not asyncio.run(broker.authorize_connect_destination(host="docs.example.com", port=443))
    assert not asyncio.run(
        broker.authorize_connect_destination(host="untrusted.example.com", port=443)
    )
    assert not asyncio.run(broker.authorize_connect_destination(host="docs.example.com", port=8443))


def test_broker_revalidates_global_response_limit_for_custom_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"bounded-upstream-response"
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=body))
    broker, registry, _resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)
    monkeypatch.setattr(
        broker_module,
        "MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES",
        len(body) - 1,
    )

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "oversized_response"
    assert upstream.sent is not None
    assert decisions[-1].allowed is False


def _request(grant_value: str, path: str, form: dict[str, str] | None = None) -> CapturedRequest:
    body = urlencode(form).encode() if form else b""
    return CapturedRequest(
        method="POST",
        host="api.stripe.com",
        path=path,
        headers={
            "Authorization": f"Bearer {grant_value}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body=body,
    )


def _no_real_secret(decisions: list[EgressDecision]) -> None:
    for decision in decisions:
        assert REAL_SECRET not in str(asdict(decision))


def _exception_graph(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    observed: list[BaseException] = []
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        observed.append(candidate)
        if isinstance(candidate, BaseExceptionGroup):
            pending.extend(candidate.exceptions)
        if candidate.__cause__ is not None:
            pending.append(candidate.__cause__)
        if candidate.__context__ is not None:
            pending.append(candidate.__context__)
    return observed


def test_revocation_before_prepared_upstream_dispatch_never_starts_operation() -> None:
    async def run() -> tuple[CapturedResponse, int, list[str | None]]:
        operation_prepared = asyncio.Event()
        result_entered = asyncio.Event()
        allow_result_start = asyncio.Event()
        prestart_cancelled = asyncio.Event()
        dispatches: list[str | None] = []

        class _PausedStartOperation(EgressUpstreamOperation):
            async def result(self) -> CapturedResponse:
                result_entered.set()
                await allow_result_start.wait()
                return await super().result()

            async def cancel_and_wait(self) -> None:
                await super().cancel_and_wait()
                prestart_cancelled.set()

        class _PausedStartUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert limits.max_response_bytes > 0

                async def send() -> CapturedResponse:
                    dispatches.append(request.headers.get("Authorization"))
                    return CapturedResponse(status_code=200, body=b"late")

                operation = _PausedStartOperation(send)
                operation_prepared.set()
                return operation

        broker, registry, _resolver, _decisions = _build(upstream=_PausedStartUpstream())
        grant = _mint(registry)
        request_task = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        revocation_task: asyncio.Task[int] | None = None
        try:
            await asyncio.wait_for(operation_prepared.wait(), timeout=1.0)
            await asyncio.wait_for(result_entered.wait(), timeout=1.0)
            revocation_task = asyncio.create_task(
                broker.revoke_authority_and_wait((grant.presented_value,))
            )
            await asyncio.wait_for(prestart_cancelled.wait(), timeout=1.0)
            assert dispatches == []

            allow_result_start.set()
            response = await asyncio.wait_for(request_task, timeout=1.0)
            revoked = await asyncio.wait_for(revocation_task, timeout=1.0)
            return response, revoked, dispatches
        finally:
            allow_result_start.set()
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            if revocation_task is not None:
                if not revocation_task.done():
                    revocation_task.cancel()
                await asyncio.gather(revocation_task, return_exceptions=True)

    response, revoked, dispatches = asyncio.run(run())

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "fetch_failed"
    assert revoked == 1
    assert dispatches == []


def test_revocation_scheduled_during_resolution_rejects_before_factory_entry() -> None:
    async def run() -> tuple[CapturedResponse, int, list[str | None]]:
        dispatches: list[str | None] = []

        class _NoCancellationOwnerUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert limits.max_response_bytes > 0

                async def send() -> CapturedResponse:
                    dispatches.append(request.headers.get("Authorization"))
                    return CapturedResponse(status_code=200, body=b"late")

                return EgressUpstreamOperation(send)

        class _RevocationSchedulingResolver:
            def __init__(self) -> None:
                self.broker: TransparentEgressBroker | None = None
                self.presented_value: str | None = None
                self.revocation_task: asyncio.Task[int] | None = None

            async def resolve(
                self,
                ref: SecretRef,
                *,
                scope: dict[str, Any] | None = None,
            ) -> ResolvedSecret:
                assert scope is not None
                assert self.broker is not None
                assert self.presented_value is not None
                self.revocation_task = asyncio.create_task(
                    self.broker.revoke_authority_and_wait((self.presented_value,))
                )
                return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))

        registry = VirtualCredentialRegistry()
        resolver = _RevocationSchedulingResolver()
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=resolver,
            policies={"stripe-example": _stripe_example_policy()},
            upstream=_NoCancellationOwnerUpstream(),
        )
        grant = _mint(registry)
        resolver.broker = broker
        resolver.presented_value = grant.presented_value

        response = await broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        assert resolver.revocation_task is not None
        revoked = await resolver.revocation_task
        return response, revoked, dispatches

    response, revoked, dispatches = asyncio.run(run())

    assert response.status_code == 403
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "request_denied"
    assert revoked == 1
    assert dispatches == []


def test_revocation_arms_every_upstream_settlement_before_waiting() -> None:
    async def run() -> tuple[list[CapturedResponse], int, list[bool]]:
        started = (asyncio.Event(), asyncio.Event())
        cancellation_requested = (asyncio.Event(), asyncio.Event())
        release_first_cancellation = asyncio.Event()

        class _TwoOperationUpstream:
            def __init__(self) -> None:
                self.prepared = 0

            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert isinstance(request, CapturedRequest)
                assert limits.max_response_bytes > 0
                index = self.prepared
                self.prepared += 1

                async def send() -> CapturedResponse:
                    started[index].set()
                    await asyncio.Event().wait()
                    raise AssertionError("Stalled upstream completed unexpectedly.")

                async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                    cancellation_requested[index].set()
                    if index == 0:
                        await release_first_cancellation.wait()
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        upstream = _TwoOperationUpstream()
        broker, registry, _resolver, _decisions = _build(upstream=upstream)
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        requests = [asyncio.create_task(broker.handle_request(request))]
        revocation_task: asyncio.Task[int] | None = None
        try:
            await asyncio.wait_for(started[0].wait(), timeout=1.0)
            requests.append(asyncio.create_task(broker.handle_request(request)))
            await asyncio.wait_for(started[1].wait(), timeout=1.0)

            revocation_task = asyncio.create_task(
                broker.revoke_authority_and_wait((grant.presented_value,))
            )
            await asyncio.wait_for(cancellation_requested[0].wait(), timeout=1.0)
            await asyncio.wait_for(cancellation_requested[1].wait(), timeout=1.0)
            assert revocation_task.done() is False

            release_first_cancellation.set()
            responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=1.0)
            revoked = await asyncio.wait_for(revocation_task, timeout=1.0)
            return responses, revoked, [event.is_set() for event in cancellation_requested]
        finally:
            release_first_cancellation.set()
            for request_task in requests:
                if not request_task.done():
                    request_task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
            if revocation_task is not None:
                if not revocation_task.done():
                    revocation_task.cancel()
                await asyncio.gather(revocation_task, return_exceptions=True)

    responses, revoked, cancellation_requests = asyncio.run(run())

    assert [response.status_code for response in responses] == [502, 502]
    assert [response.headers[CAYU_EGRESS_ERROR_HEADER] for response in responses] == [
        "fetch_failed",
        "fetch_failed",
    ]
    assert revoked == 1
    assert cancellation_requests == [True, True]


@pytest.mark.parametrize("signal_type", [GeneratorExit, KeyboardInterrupt, SystemExit])
def test_revocation_preserves_upstream_settlement_process_control_signal(
    signal_type: type[BaseException],
) -> None:
    fatal_signal = signal_type("upstream settlement interrupted")

    async def run() -> tuple[BaseExceptionGroup, int]:
        started = asyncio.Event()
        cancel_calls = 0

        class _FatalSettlementUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert isinstance(request, CapturedRequest)
                assert limits.max_response_bytes > 0

                async def send() -> CapturedResponse:
                    started.set()
                    await asyncio.Event().wait()
                    raise AssertionError("Stalled upstream completed unexpectedly.")

                async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                    nonlocal cancel_calls
                    cancel_calls += 1
                    if cancel_calls == 1:
                        raise fatal_signal
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        broker, registry, _resolver, _decisions = _build(upstream=_FatalSettlementUpstream())
        grant = _mint(registry)
        request_task = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        failure: BaseExceptionGroup | None = None
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await broker.revoke_authority_and_wait((grant.presented_value,))
            failure = exc_info.value
        finally:
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            await broker.settle_active_upstream_operations()
        assert failure is not None
        return failure, cancel_calls

    failure, cancel_calls = asyncio.run(run())

    observed = _exception_graph(failure)
    assert sum(candidate is fatal_signal for candidate in observed) == 1
    assert cancel_calls == 2


def test_caller_cancellation_preserves_grouped_upstream_process_control_signal() -> None:
    fatal_signal = GeneratorExit("upstream settlement interrupted")
    settlement_failure = BaseExceptionGroup(
        "upstream settlement failed",
        [
            RuntimeError("ordinary cleanup failure"),
            BaseExceptionGroup("nested fatal signal", [fatal_signal]),
        ],
    )

    async def run() -> tuple[BaseExceptionGroup, bool, int, int]:
        started = asyncio.Event()
        cancel_calls = 0

        class _FatalSettlementUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert isinstance(request, CapturedRequest)
                assert limits.max_response_bytes > 0

                async def send() -> CapturedResponse:
                    started.set()
                    await asyncio.Event().wait()
                    raise AssertionError("Stalled upstream completed unexpectedly.")

                async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                    nonlocal cancel_calls
                    cancel_calls += 1
                    if cancel_calls == 1:
                        raise settlement_failure
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        broker, registry, _resolver, _decisions = _build(upstream=_FatalSettlementUpstream())
        grant = _mint(registry)
        request_task = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        failure: BaseExceptionGroup | None = None
        cancelled = False
        cancelling = 0
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            request_task.cancel("caller disconnected")
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await request_task
            failure = exc_info.value
            cancelled = request_task.cancelled()
            cancelling = request_task.cancelling()
        finally:
            await broker.settle_active_upstream_operations()
        assert failure is not None
        return failure, cancelled, cancelling, cancel_calls

    failure, cancelled, cancelling, cancel_calls = asyncio.run(run())

    observed = _exception_graph(failure)
    assert sum(candidate is settlement_failure for candidate in observed) == 1
    assert sum(candidate is fatal_signal for candidate in observed) == 1
    cancellations = [
        candidate for candidate in observed if isinstance(candidate, asyncio.CancelledError)
    ]
    assert len(cancellations) == 1
    assert str(cancellations[0]) == "caller disconnected"
    assert cancelled is False
    assert cancelling == 1
    assert cancel_calls == 2


def test_caller_cancellation_releases_quiescent_process_control_capacity() -> None:
    fatal_signal = GeneratorExit("upstream operation interrupted")

    async def run() -> tuple[GeneratorExit, CapturedResponse, bool, int, int]:
        started = asyncio.Event()
        prepared = 0

        class _FatalOperationUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                nonlocal prepared
                assert isinstance(request, CapturedRequest)
                assert limits.max_response_bytes > 0
                prepared += 1
                operation_number = prepared

                async def send() -> CapturedResponse:
                    if operation_number > 1:
                        return CapturedResponse(status_code=200, body=b"recovered")
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError as cancellation:
                        raise fatal_signal from cancellation
                    raise AssertionError("Stalled upstream completed unexpectedly.")

                async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        # The cancellation owner has positively observed task
                        # quiescence. The broker must independently preserve
                        # the operation's authoritative outcome.
                        return

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        broker, registry, _resolver, _decisions = _build(
            upstream=_FatalOperationUpstream(),
            max_active_upstream_operations=1,
        )
        first_grant = _mint(registry)
        request_task = asyncio.create_task(
            broker.handle_request(_request(first_grant.presented_value, "/v1/customers"))
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        request_task.cancel("caller disconnected")
        with pytest.raises(GeneratorExit) as exc_info:
            await request_task

        replacement = _mint(registry)
        response = await broker.handle_request(
            _request(replacement.presented_value, "/v1/customers")
        )
        await broker.settle_active_operations()
        return (
            exc_info.value,
            response,
            request_task.cancelled(),
            request_task.cancelling(),
            prepared,
        )

    failure, response, cancelled, cancelling, prepared = asyncio.run(run())

    observed = _exception_graph(failure)
    assert sum(candidate is fatal_signal for candidate in observed) == 1
    cancellations = [
        candidate for candidate in observed if isinstance(candidate, asyncio.CancelledError)
    ]
    assert len(cancellations) == 1
    assert str(cancellations[0]) == "caller disconnected"
    assert response.status_code == 200
    assert response.body == b"recovered"
    assert cancelled is False
    assert cancelling == 1
    assert prepared == 2


def test_allowed_request_injects_real_secret_upstream_only() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    # Upstream received the REAL secret...
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"
    # ...and the virtual value never went upstream.
    assert grant.presented_value not in str(upstream.sent.headers)
    # Response returned to the sandbox contains no real secret.
    assert REAL_SECRET not in response.body.decode()
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    _no_real_secret(decisions)


def test_stripe_basic_request_injects_real_secret_as_bearer_upstream() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)
    basic_value = base64.b64encode(f"{grant.presented_value}:".encode()).decode()
    request = CapturedRequest(
        method="POST",
        host="api.stripe.com",
        path="/v1/customers",
        headers={
            "Authorization": f"Basic {basic_value}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"
    assert grant.presented_value not in str(upstream.sent.headers)
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    _no_real_secret(decisions)


def test_opaque_token_request_injects_real_secret_upstream_only() -> None:
    upstream = _FakeUpstream(
        CapturedResponse(
            status_code=200,
            body=f'{{"debug":"{GITHUB_SECRET}"}}'.encode(),
        )
    )
    github_policy = HttpEgressPolicy(
        name="github-read",
        allowed_hosts=["api.github.com"],
        allowed_endpoints=[("GET", "/user")],
    )
    broker, registry, resolver, decisions = _build(
        upstream=upstream,
        policies={"github-read": github_policy},
    )
    grant = _mint(
        registry,
        env_name="GH_TOKEN",
        secret=SecretRef(name="github_token"),
        destination="api.github.com",
        credential_kind="opaque_token",
        policy_name="github-read",
    )
    request = CapturedRequest(
        method="GET",
        host="api.github.com",
        path="/user",
        headers={"Authorization": f"token {grant.presented_value}"},
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"token {GITHUB_SECRET}"
    assert grant.presented_value not in str(upstream.sent.headers)
    assert GITHUB_SECRET not in response.body.decode()
    assert b"[REDACTED_SECRET]" in response.body
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    assert GITHUB_SECRET not in str(asdict(decisions[-1]))
    _no_real_secret(decisions)


@pytest.mark.parametrize(
    ("credential_kind", "authorization_template"),
    [
        ("opaque_bearer", "token {credential}"),
        ("opaque_token", "Bearer {credential}"),
        ("opaque_token", "{credential}"),
    ],
)
def test_opaque_credential_rejects_mismatched_or_missing_authorization_scheme(
    credential_kind: str,
    authorization_template: str,
) -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    github_policy = HttpEgressPolicy(
        name="github-read",
        allowed_hosts=["api.github.com"],
        allowed_endpoints=[("GET", "/user")],
    )
    broker, registry, resolver, decisions = _build(
        upstream=upstream,
        policies={"github-read": github_policy},
    )
    grant = _mint(
        registry,
        env_name="GH_TOKEN",
        secret=SecretRef(name="github_token"),
        destination="api.github.com",
        credential_kind=credential_kind,
        policy_name="github-read",
    )
    request = CapturedRequest(
        method="GET",
        host="api.github.com",
        path="/user",
        headers={"Authorization": authorization_template.format(credential=grant.presented_value)},
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 403
    assert b"authentication scheme does not match" in response.body
    assert resolver.resolve_count == 0
    assert upstream.sent is None
    assert decisions[-1].allowed is False


def test_allowed_response_redacts_echoed_real_secret() -> None:
    upstream = _FakeUpstream(
        CapturedResponse(
            status_code=200,
            headers={"X-Echo-Secret": f"provider echoed {REAL_SECRET}"},
            body=f'{{"debug":"{REAL_SECRET}"}}'.encode(),
        )
    )
    broker, registry, _resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert REAL_SECRET not in str(response.headers)
    assert REAL_SECRET not in response.body.decode()
    assert "[REDACTED_SECRET]" in response.headers["X-Echo-Secret"]
    assert b"[REDACTED_SECRET]" in response.body
    _no_real_secret(decisions)


def test_response_redaction_cannot_expand_body_beyond_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_secret = "sk_test_x"
    body = short_secret.encode() * 4
    monkeypatch.setattr(
        broker_module,
        "MAX_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES",
        len(body),
    )
    registry = VirtualCredentialRegistry()
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": short_secret}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=_FakeUpstream(CapturedResponse(status_code=200, body=body)),
        audit=decisions.append,
        browser_max_response_bytes=len(body),
    )
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "oversized_response"
    assert short_secret.encode() not in response.body
    assert decisions[-1].allowed is False


def test_denied_endpoint_never_resolves_secret() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/payouts", {"amount": "100"}))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 0  # deny-before-resolve
    assert upstream.sent is None  # never forwarded
    assert decisions[-1].allowed is False
    _no_real_secret(decisions)


def test_unknown_credential_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request("sk_test_cayu_vc_bogus", "/v1/customers"))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 0
    _no_real_secret(decisions)


def test_missing_credential_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, _registry, resolver, _decisions = _build(upstream=upstream)

    request = CapturedRequest(method="GET", host="api.stripe.com", path="/v1/customers")
    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 401
    assert resolver.resolve_count == 0


def test_expired_credential_denied() -> None:
    clock = _Clock(datetime(2026, 7, 6, tzinfo=UTC))
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream, clock=clock)
    grant = _mint(registry, ttl_seconds=60)
    clock.advance(61)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_destination_mismatch_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream)
    grant = _mint(registry, destination="api.stripe.com")

    # A request whose host differs from the grant binding.
    request = CapturedRequest(
        method="POST",
        host="uploads.stripe.com",
        path="/v1/customers",
        headers={"Authorization": f"Bearer {grant.presented_value}"},
    )
    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_missing_policy_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream, policies={})
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_unsupported_credential_kind_rejected_at_mint() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream)

    with pytest.raises(ValueError, match="Unsupported credential kind"):
        _mint(registry, credential_kind="mystery_kind")
    assert resolver.resolve_count == 0
    assert broker.registry is registry


def test_upstream_failure_is_sanitized() -> None:
    broker, registry, _resolver, decisions = _build(upstream=_FailingUpstream())
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "fetch_failed"
    assert REAL_SECRET not in response.body.decode()
    _no_real_secret(decisions)


def test_successful_upstream_cannot_spoof_internal_broker_diagnostic() -> None:
    upstream = _FakeUpstream(
        CapturedResponse(
            status_code=200,
            headers={CAYU_EGRESS_ERROR_HEADER: "destination_denied"},
            body=b"{}",
        )
    )
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 200
    assert CAYU_EGRESS_ERROR_HEADER not in response.headers


def test_httpx_upstream_rejects_compression_before_decoded_body_allocation() -> None:
    body_started = False
    captured: list[httpx.Request] = []

    class _CompressedBody(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            nonlocal body_started
            body_started = True
            yield gzip.compress(b"x" * (8 * 1024 * 1024 + 1))

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
            stream=_CompressedBody(),
            request=request,
        )

    async def run() -> CapturedResponse:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )
        return await upstream.send(
            CapturedRequest(method="GET", host="api.stripe.com", path="/v1/customers")
        )

    with pytest.raises(RuntimeError, match="ignored the required identity"):
        asyncio.run(run())

    assert body_started is False
    assert str(captured[0].url) == "https://93.184.216.34/v1/customers"
    assert captured[0].headers["host"] == "api.stripe.com"
    assert captured[0].headers["accept-encoding"] == "identity"
    assert captured[0].extensions["sni_hostname"] == "api.stripe.com"


def test_httpx_upstream_accepts_response_at_exact_decoded_byte_limit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"bounded", request=request)

    async def run() -> CapturedResponse:
        upstream = HttpxUpstream(
            max_response_bytes=len(b"bounded"),
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )
        return await upstream.send(
            CapturedRequest(method="GET", host="api.stripe.com", path="/v1/customers")
        )

    response = asyncio.run(run())

    assert response.body == b"bounded"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_response_bytes": True}, TypeError),
        ({"max_response_bytes": 0}, ValueError),
        ({"max_response_bytes": 64 * 1024 * 1024 + 1}, ValueError),
        ({"timeout_s": True}, TypeError),
        ({"timeout_s": 0}, ValueError),
        ({"timeout_s": float("nan")}, ValueError),
    ],
)
def test_httpx_upstream_rejects_unbounded_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        HttpxUpstream(**kwargs)


def test_httpx_upstream_does_not_forward_reserved_broker_diagnostic() -> None:
    captured: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = request
        return httpx.Response(200, content=b"{}", request=request)

    async def run() -> None:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )
        await upstream.send(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/",
                headers={CAYU_EGRESS_ERROR_HEADER: "destination_denied"},
            )
        )

    asyncio.run(run())

    assert captured is not None
    assert CAYU_EGRESS_ERROR_HEADER.lower() not in captured.headers


def test_broker_rejects_encoded_upstream_response_before_body_read() -> None:
    decoded = b"decoded response exceeds the bound"
    encoded = gzip.compress(decoded)
    body_started = False

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            nonlocal body_started
            body_started = True
            yield encoded

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=_Body(),
            request=request,
        )

    upstream = HttpxUpstream(
        max_response_bytes=len(decoded) - 1,
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    broker, registry, _resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "unsupported_content"
    assert decisions[-1].reason == "Upstream ignored the required identity content encoding."
    assert body_started is False


def test_browser_policy_requires_identity_encoding_before_upstream_body_read() -> None:
    body_started = False
    accepted_encoding: str | None = None

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            nonlocal body_started
            body_started = True
            yield gzip.compress(b"expanded browser content")

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal accepted_encoding
        accepted_encoding = request.headers.get("accept-encoding")
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=_Body(),
            request=request,
        )

    upstream = HttpxUpstream(
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    broker = TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=None,
        policies={
            "browser": BrowserEgressPolicy(
                name="browser",
                allowed_hosts=["docs.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="browser",
            )
        ],
        upstream=upstream,
    )

    response = asyncio.run(
        broker.handle_request(CapturedRequest(method="GET", host="docs.example.com", path="/"))
    )

    assert accepted_encoding == "identity"
    assert body_started is False
    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "unsupported_content"


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            CapturedResponse(
                status_code=200,
                headers={"Content-Encoding": "gzip"},
                body=b"compressed",
            ),
            "unsupported_content",
        ),
        (
            CapturedResponse(status_code=200, body=b"too large"),
            "oversized_response",
        ),
    ],
)
def test_browser_policy_revalidates_custom_upstream_response_contract(
    monkeypatch: pytest.MonkeyPatch,
    response: CapturedResponse,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        "cayu.egress.broker.DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES",
        4,
    )
    broker = TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=None,
        policies={
            "browser": BrowserEgressPolicy(
                name="browser",
                allowed_hosts=["docs.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="browser",
            )
        ],
        upstream=_FakeUpstream(response),
    )

    result = asyncio.run(
        broker.handle_request(CapturedRequest(method="GET", host="docs.example.com", path="/"))
    )

    assert result.status_code == 502
    assert result.headers[CAYU_EGRESS_ERROR_HEADER] == expected_error


def test_browser_policy_uses_configured_response_limit_for_custom_upstream() -> None:
    body = b"larger than the default fixture limit"
    broker = TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=None,
        policies={
            "browser": BrowserEgressPolicy(
                name="browser",
                allowed_hosts=["files.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="files.example.com",
                policy_name="browser",
            )
        ],
        upstream=_FakeUpstream(CapturedResponse(status_code=200, body=body)),
        browser_max_response_bytes=len(body),
    )

    result = asyncio.run(
        broker.handle_request(CapturedRequest(method="GET", host="files.example.com", path="/"))
    )

    assert result.status_code == 200
    assert result.body == body


def test_broker_rejects_announced_upstream_response_over_byte_limit_before_read() -> None:
    body_started = False

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            nonlocal body_started
            body_started = True
            yield b"unexpected"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "100"},
            stream=_Body(),
            request=request,
        )

    upstream = HttpxUpstream(
        max_response_bytes=10,
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "oversized_response"
    assert body_started is False


def test_broker_classifies_upstream_dns_failure_without_contacting_transport() -> None:
    transport_called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_called
        transport_called = True
        return httpx.Response(200, request=request)

    upstream = HttpxUpstream(
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver(),
    )
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "dns_failure"
    assert transport_called is False


def test_broker_classifies_prohibited_upstream_destination() -> None:
    upstream = HttpxUpstream(destination_resolver=_destination_resolver("127.0.0.1"))
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 403
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "destination_denied"


def test_broker_classifies_upstream_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow upstream", request=request)

    upstream = HttpxUpstream(
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 504
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "timeout"


def test_httpx_upstream_enforces_total_deadline_against_slow_drip_response() -> None:
    closed = False

    class _SlowDripBody(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for _ in range(10):
                await asyncio.sleep(0.03)
                yield b"x"

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_SlowDripBody(), request=request)

    upstream = HttpxUpstream(
        timeout_s=0.05,
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    started = time.monotonic()
    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert time.monotonic() - started < 0.5
    assert response.status_code == 504
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "timeout"
    assert closed is True
    assert registry._active_counts == {}


def test_injected_transport_remains_fenced_until_opaque_dispatch_settles() -> None:
    entered = threading.Event()
    release = threading.Event()
    dispatches = 0

    def blocking_transport() -> None:
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("Injected transport test operation was not released.")

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await asyncio.to_thread(blocking_transport)
        return httpx.Response(200, content=b"{}", request=request)

    async def run() -> tuple[CapturedResponse, CapturedResponse, bool, int]:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            routes={"api.stripe.com": "https://93.184.216.34"},
        )
        broker, registry, _resolver, _decisions = _build(
            upstream=upstream,
            max_active_upstream_operations=1,
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        first = asyncio.create_task(broker.handle_request(request))
        try:
            assert await asyncio.to_thread(entered.wait, 2.0)
            first.cancel("caller stopped waiting")
            await asyncio.sleep(0.05)
            assert first.done() is False

            exhausted = await broker.handle_request(request)
            release.set()
            with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
                await first
            recovered = await broker.handle_request(request)
            return exhausted, recovered, first.cancelled(), first.cancelling()
        finally:
            release.set()
            if not first.done():
                with contextlib.suppress(BaseException):
                    await first

    exhausted, recovered, cancelled, cancelling = asyncio.run(run())

    assert exhausted.status_code == 503
    assert exhausted.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"
    assert recovered.status_code == 200
    assert cancelled is True
    assert cancelling == 1
    assert dispatches == 2


def test_cancelled_credential_resolution_retains_capacity_until_thread_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class _OpaqueResolver:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            del scope
            self.calls += 1
            if self.calls == 1:

                def blocking_read() -> None:
                    entered.set()
                    if not release.wait(5.0):
                        raise TimeoutError("Opaque credential resolution was not released.")

                await asyncio.to_thread(blocking_read)
            return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))

    async def run() -> tuple[CapturedResponse, CapturedResponse, bool, int, int, int]:
        resolver = _OpaqueResolver()
        registry = VirtualCredentialRegistry()
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=resolver,
            policies={"stripe-example": _stripe_example_policy()},
            upstream=_FakeUpstream(CapturedResponse(status_code=200, body=b"{}")),
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        first = asyncio.create_task(broker.handle_request(request))
        revocation: asyncio.Task[int] | None = None
        try:
            assert await asyncio.to_thread(entered.wait, 2.0)
            first.cancel("caller disconnected")
            await asyncio.sleep(0.05)
            assert first.done() is False

            exhausted = await broker.handle_request(request)
            revocation = asyncio.create_task(
                broker.revoke_authority_and_wait((grant.presented_value,))
            )
            await asyncio.sleep(0.05)
            assert revocation.done() is False

            release.set()
            with pytest.raises(asyncio.CancelledError, match="caller disconnected"):
                await first
            revoked = await asyncio.wait_for(revocation, timeout=1.0)
            replacement = _mint(registry)
            recovered = await broker.handle_request(
                _request(replacement.presented_value, "/v1/customers")
            )
            return (
                exhausted,
                recovered,
                first.cancelled(),
                first.cancelling(),
                resolver.calls,
                revoked,
            )
        finally:
            release.set()
            if not first.done():
                with contextlib.suppress(BaseException):
                    await first
            if revocation is not None and not revocation.done():
                await revocation

    monkeypatch.setattr(
        broker_module,
        "_MAX_ACTIVE_CREDENTIAL_RESOLUTIONS",
        1,
    )
    exhausted, recovered, cancelled, cancelling, calls, revoked = asyncio.run(run())

    assert exhausted.status_code == 503
    assert exhausted.headers[CAYU_EGRESS_ERROR_HEADER] == "credential_resolution_capacity_exhausted"
    assert recovered.status_code == 200
    assert cancelled is True
    assert cancelling == 1
    assert calls == 2
    assert revoked == 1


def test_credential_resolution_settlement_cancels_all_queued_work_before_waiting() -> None:
    async def run() -> tuple[bool, bool]:
        broker, _registry, _resolver, _decisions = _build(
            upstream=_FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
        )
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        allow_second = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_resolution() -> ResolvedSecret:
            first_entered.set()
            await release_first.wait()
            return ResolvedSecret(name="stripe_test_key", value=SecretStr(REAL_SECRET))

        async def queued_resolution() -> ResolvedSecret:
            await allow_second.wait()
            second_entered.set()
            return ResolvedSecret(name="stripe_test_key", value=SecretStr(REAL_SECRET))

        first_task = asyncio.create_task(first_resolution())
        await first_entered.wait()
        second_task = asyncio.create_task(queued_resolution())
        first_token = object()
        second_token = object()
        broker._active_credential_resolutions = {
            first_token: broker_module._ActiveCredentialResolution(
                task=first_task,
                started=True,
            ),
            second_token: broker_module._ActiveCredentialResolution(
                task=second_task,
                started=False,
            ),
        }
        broker._credential_resolutions_idle.clear()
        settlement = asyncio.create_task(broker.settle_active_credential_resolutions())
        try:
            await asyncio.sleep(0)
            allow_second.set()
            await asyncio.sleep(0)
            queued_was_cancelled = second_task.cancelled()
            entered_before_first_settled = second_entered.is_set()
            release_first.set()
            await settlement
            return queued_was_cancelled, entered_before_first_settled
        finally:
            release_first.set()
            allow_second.set()
            if not settlement.done():
                settlement.cancel()
            await asyncio.gather(settlement, first_task, second_task, return_exceptions=True)

    queued_was_cancelled, entered_before_first_settled = asyncio.run(run())

    assert queued_was_cancelled is True
    assert entered_before_first_settled is False


def test_child_credential_resolution_cancellation_is_diagnostically_sanitized(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "CREDENTIAL_CANCELLATION_SECRET_CANARY"

    class _CancellingResolver:
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            del ref, scope
            raise asyncio.CancelledError(canary)

    async def run() -> tuple[RuntimeError, CapturedResponse, list[EgressDecision]]:
        registry = VirtualCredentialRegistry()
        decisions: list[EgressDecision] = []
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=_CancellingResolver(),
            policies={"stripe-example": _stripe_example_policy()},
            upstream=_FakeUpstream(CapturedResponse(status_code=200, body=b"{}")),
            audit=decisions.append,
        )
        grant = _mint(registry)
        with pytest.raises(
            RuntimeError,
            match="Credential resolution stopped without caller cancellation",
        ) as excinfo:
            await broker._resolve_credential(grant.secret, grant_id=grant.grant_id)
        response = await broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        return excinfo.value, response, decisions

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        error, response, decisions = asyncio.run(run())

    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            repr(error),
            repr(response),
            repr(decisions),
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
            *(record.getMessage() for record in caplog.records),
        ]
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    assert response.status_code == 502
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "request_denied"
    assert caught == []
    assert canary not in diagnostics


def test_injected_transport_timeout_remains_fenced_until_opaque_dispatch_settles() -> None:
    entered = threading.Event()
    release = threading.Event()
    dispatches = 0

    def blocking_transport() -> None:
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("Injected transport test operation was not released.")

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        if dispatches == 1:
            await asyncio.to_thread(blocking_transport)
        return httpx.Response(200, content=b"{}", request=request)

    async def run() -> tuple[CapturedResponse, CapturedResponse, CapturedResponse]:
        upstream = HttpxUpstream(
            timeout_s=0.05,
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )
        broker, registry, _resolver, _decisions = _build(
            upstream=upstream,
            max_active_upstream_operations=1,
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        first = asyncio.create_task(broker.handle_request(request))
        try:
            assert await asyncio.to_thread(entered.wait, 2.0)
            await asyncio.sleep(0.1)
            assert first.done() is False
            exhausted = await broker.handle_request(request)
            release.set()
            timed_out = await first
            recovered = await broker.handle_request(request)
            return exhausted, timed_out, recovered
        finally:
            release.set()
            if not first.done():
                with contextlib.suppress(BaseException):
                    await first

    exhausted, timed_out, recovered = asyncio.run(run())

    assert exhausted.status_code == 503
    assert exhausted.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"
    assert timed_out.status_code == 504
    assert timed_out.headers[CAYU_EGRESS_ERROR_HEADER] == "timeout"
    assert recovered.status_code == 200
    assert dispatches == 2


def test_default_dns_deadline_kills_owned_resolver_before_releasing_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "blocked_resolver.py"
    helper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    spawned: list[asyncio.subprocess.Process] = []
    original_spawn = asyncio.create_subprocess_exec

    async def recording_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await original_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        return process

    monkeypatch.setattr(resolution_module, "_OWNED_RESOLVER_HELPER", helper)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)
    transport_called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_called
        transport_called = True
        return httpx.Response(200, content=b"{}", request=request)

    upstream = HttpxUpstream(
        timeout_s=0.05,
        transport=httpx.MockTransport(handler),
        routes={"api.stripe.com": "https://93.184.216.34"},
    )
    broker, registry, _resolver, _decisions = _build(
        upstream=upstream,
        max_active_upstream_operations=1,
    )
    grant = _mint(registry)

    started = time.monotonic()
    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert time.monotonic() - started < 1.0
    assert response.status_code == 504
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "timeout"
    assert transport_called is False
    assert len(spawned) == 1
    assert spawned[0].returncode is not None
    assert broker._active_upstream_operations == {}
    assert registry._active_counts == {}


def test_dns_cancellation_retains_capacity_when_helper_kill_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entered = tmp_path / "resolver-entered"
    release = tmp_path / "resolver-release"
    helper = tmp_path / "blocked_resolver.py"
    helper.write_text(
        "from pathlib import Path\n"
        "import time\n"
        f"entered = Path({str(entered)!r})\n"
        f"release = Path({str(release)!r})\n"
        "entered.touch()\n"
        "while not release.exists():\n"
        "    time.sleep(0.01)\n"
        "print('[\"93.184.216.34\"]')\n",
        encoding="utf-8",
    )
    spawned: list[asyncio.subprocess.Process] = []
    original_spawn = asyncio.create_subprocess_exec
    original_kill = asyncio.subprocess.Process.kill

    async def recording_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await original_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        return process

    def fail_first_helper_kill(process: asyncio.subprocess.Process) -> None:
        if spawned and process is spawned[0]:
            raise PermissionError("resolver termination denied")
        original_kill(process)

    monkeypatch.setattr(resolution_module, "_OWNED_RESOLVER_HELPER", helper)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)
    monkeypatch.setattr(asyncio.subprocess.Process, "kill", fail_first_helper_kill)
    transport_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(200, content=b"{}", request=request)

    async def run() -> tuple[CapturedResponse, CapturedResponse, bool, int]:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            routes={"api.stripe.com": "https://93.184.216.34"},
        )
        broker, registry, _resolver, _decisions = _build(
            upstream=upstream,
            max_active_upstream_operations=1,
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        first = asyncio.create_task(broker.handle_request(request))
        try:
            for _attempt in range(200):
                if entered.exists():
                    break
                await asyncio.sleep(0.01)
            assert entered.exists()
            first.cancel("caller stopped waiting")
            await asyncio.sleep(0.05)

            assert first.done() is False
            assert spawned[0].returncode is None
            assert broker._active_upstream_operations
            exhausted = await broker.handle_request(request)

            release.touch()
            with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
                await first
            recovered = await broker.handle_request(request)
            return exhausted, recovered, first.cancelled(), first.cancelling()
        finally:
            release.touch(exist_ok=True)
            if not first.done():
                with contextlib.suppress(BaseException):
                    await first

    exhausted, recovered, cancelled, cancelling = asyncio.run(run())

    assert exhausted.status_code == 503
    assert exhausted.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"
    assert recovered.status_code == 200
    assert cancelled is True
    assert cancelling == 1
    assert transport_calls == 1
    assert spawned[0].returncode is not None


def test_dns_cancellation_retains_upstream_capacity_until_opaque_resolution_settles() -> None:
    entered = threading.Event()
    release = threading.Event()
    transport_calls = 0

    def blocking_resolution() -> tuple[str, ...]:
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("DNS test operation was not released.")
        return ("93.184.216.34",)

    async def resolver(_host: str, _port: int) -> tuple[str, ...]:
        return await asyncio.to_thread(blocking_resolution)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(200, content=b"{}", request=request)

    async def run() -> tuple[CapturedResponse, CapturedResponse, bool, int]:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=resolver,
        )
        broker, registry, _resolver, _decisions = _build(
            upstream=upstream,
            max_active_upstream_operations=1,
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")
        first = asyncio.create_task(broker.handle_request(request))
        try:
            assert await asyncio.to_thread(entered.wait, 2.0)
            first.cancel("caller stopped waiting")
            await asyncio.sleep(0.05)
            assert first.done() is False
            # The settlement owner temporarily consumes the delivered request;
            # it is restored immediately before cancellation is re-delivered.
            assert first.cancelling() == 0

            exhausted = await broker.handle_request(request)
            release.set()
            with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
                await first
            recovered = await broker.handle_request(request)
            return exhausted, recovered, first.cancelled(), first.cancelling()
        finally:
            release.set()
            if not first.done():
                with contextlib.suppress(BaseException):
                    await first

    exhausted, recovered, cancelled, cancelling = asyncio.run(run())

    assert exhausted.status_code == 503
    assert exhausted.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"
    assert recovered.status_code == 200
    assert cancelled is True
    assert cancelling == 1
    assert transport_calls == 1


def test_httpx_upstream_routes_logical_host_to_private_service() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(202, json={"accepted": True}, request=request)

    async def run() -> CapturedResponse:
        upstream = HttpxUpstream(
            routes={"RECEIVER.Internal.": "http://receiver.service.local:8080"},
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("10.0.0.10"),
        )
        return await upstream.send(
            CapturedRequest(
                method="POST",
                host="receiver.internal",
                path="/v1/actions",
                query="mode=safe",
                headers={"Authorization": "Bearer real-secret"},
                body=b"{}",
            )
        )

    response = asyncio.run(run())

    assert response.status_code == 202
    assert str(captured[0].url) == "http://10.0.0.10:8080/v1/actions?mode=safe"
    assert captured[0].headers["host"] == "receiver.service.local:8080"
    assert captured[0].headers["authorization"] == "Bearer real-secret"


@pytest.mark.parametrize(
    "addresses",
    [
        ("169.254.169.254",),
        ("127.0.0.1",),
        ("10.0.0.1",),
        ("93.184.216.34", "169.254.169.254"),
    ],
)
def test_httpx_upstream_rejects_dns_rebinding_before_transport(
    addresses: tuple[str, ...],
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    async def run() -> None:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver(*addresses),
        )
        with pytest.raises(ValueError, match="prohibited address"):
            await upstream.send(
                CapturedRequest(
                    method="GET",
                    host="docs.example.com",
                    path="/sdk/index.json",
                )
            )

    asyncio.run(run())

    assert requests == []


@pytest.mark.parametrize(
    "route",
    [
        "receiver.service.local:8080",
        "ftp://receiver.service.local",
        "http://user:password@receiver.service.local",
        "http://receiver.service.local/base?unsafe=1",
    ],
)
def test_httpx_upstream_rejects_unsafe_private_service_route(route: str) -> None:
    with pytest.raises(ValueError, match="route"):
        HttpxUpstream(routes={"receiver.internal": route})


def test_deny_body_is_valid_json() -> None:
    import json

    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    # A denied endpoint (policy denial with a reason that contains characters
    # that would break a naive f-string body, e.g. an apostrophe).
    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/payouts", {"amount": "100"}))
    )

    assert response.status_code == 403
    decoded = json.loads(response.body)  # must be valid JSON
    assert isinstance(decoded["error"]["message"], str)


def test_audit_failure_does_not_drop_response() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    registry = VirtualCredentialRegistry()

    def _boom(_decision: EgressDecision) -> None:
        raise RuntimeError("audit sink is down")

    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=_boom,
    )
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    # The provider response survives a failing audit sink.
    assert response.status_code == 200
    assert upstream.sent is not None


class _FailingResolver:
    async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"vault down for {REAL_SECRET}")  # secret in error must not leak


class _RevokingResolver:
    def __init__(self, registry: VirtualCredentialRegistry, presented_value: str) -> None:
        self._registry = registry
        self._presented_value = presented_value
        self.resolve_count = 0

    async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
        self.resolve_count += 1
        self._registry.revoke(self._presented_value)
        return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))


class _SuspendingResolver:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del scope
        self.started.set()
        await self.resume.wait()
        return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))


def test_resolver_failure_is_labeled_distinctly() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    registry = VirtualCredentialRegistry()
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=_FailingResolver(),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=decisions.append,
    )
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 502
    assert b"Credential resolution failed" in response.body
    assert upstream.sent is None  # never reached the upstream
    assert REAL_SECRET not in response.body.decode()
    _no_real_secret(decisions)


def test_revoked_after_resolution_is_not_forwarded_upstream() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    registry = VirtualCredentialRegistry()
    grant = _mint(registry)
    resolver = _RevokingResolver(registry, grant.presented_value)
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=resolver,
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=decisions.append,
    )

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 1
    assert upstream.sent is None
    assert REAL_SECRET not in response.body.decode()
    assert decisions[-1].allowed is False
    _no_real_secret(decisions)


def test_request_mutation_during_resolution_cannot_redirect_real_credential() -> None:
    async def run() -> tuple[CapturedResponse, CapturedRequest | None]:
        upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
        registry = VirtualCredentialRegistry()
        resolver = _SuspendingResolver()
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=resolver,
            policies={"stripe-example": _stripe_example_policy()},
            upstream=upstream,
        )
        grant = _mint(registry)
        request = _request(grant.presented_value, "/v1/customers")

        pending = asyncio.create_task(broker.handle_request(request))
        await resolver.started.wait()
        request.host = "evil.example.com"
        resolver.resume.set()
        return await pending, upstream.sent

    response, sent = asyncio.run(run())

    assert response.status_code == 200
    assert sent is not None
    assert sent.host == "api.stripe.com"
    assert sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"


def test_mutated_request_validation_diagnostics_do_not_expose_hostile_repr(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "MUTATED_HOST_SECRET_CANARY"

    class HostileHost:
        def __repr__(self) -> str:
            return canary

    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)
    request = _request(grant.presented_value, "/v1/customers")
    object.__setattr__(request, "host", HostileHost())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            asyncio.run(broker.handle_request(request))

    captured = capsys.readouterr()
    diagnostics = "\n".join(
        [
            str(excinfo.value),
            captured.out,
            captured.err,
            *(str(item.message) for item in caught),
            *(record.getMessage() for record in caplog.records),
        ]
    )
    assert caught == []
    assert canary not in diagnostics
    assert len(diagnostics) < 500
    assert resolver.resolve_count == 0
    assert upstream.sent is None


LIVE_SECRET = "sk_live_51ProductionKeyBoundByMistake"


def _broker_with_secret(secret_value: str, *, upstream: Any, require_test_mode: bool = True):
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": secret_value}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        require_test_mode_credentials=require_test_mode,
    )
    return broker, registry


def test_live_key_is_refused_by_default() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry = _broker_with_secret(LIVE_SECRET, upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 403
    assert b"test-mode key" in response.body
    assert upstream.sent is None  # never forwarded a live key upstream
    assert LIVE_SECRET not in response.body.decode()


def test_test_mode_key_passes_the_guard() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry = _broker_with_secret("sk_test_51fine", upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == "Bearer sk_test_51fine"


def test_live_key_allowed_when_opted_out() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry = _broker_with_secret(LIVE_SECRET, upstream=upstream, require_test_mode=False)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {LIVE_SECRET}"


class _RotatingResolver:
    """Returns a different resolved value on each call (simulates rotation)."""

    def __init__(self, values: list[str]) -> None:
        self._values = values
        self._index = 0

    async def resolve(self, ref: SecretRef, *, scope: Any = None) -> ResolvedSecret:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return ResolvedSecret(name=ref.name, value=SecretStr(value))


class _MultiCaptureUpstream:
    def __init__(self) -> None:
        self.authorizations: list[str | None] = []

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            self.authorizations.append(request.headers.get("Authorization"))
            return CapturedResponse(status_code=200, body=b"{}")

        return EgressUpstreamOperation(send)


def test_broker_uses_rotated_secret_per_request() -> None:
    # The broker resolves the SecretRef fresh on every request, so a rotated
    # vault value takes effect immediately with no change inside the sandbox.
    upstream = _MultiCaptureUpstream()
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=_RotatingResolver(["sk_test_rotated_one", "sk_test_rotated_two"]),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
    )
    grant = _mint(registry)

    for _ in range(2):
        asyncio.run(
            broker.handle_request(
                _request(grant.presented_value, "/v1/customers", {"email": "a@b.co"})
            )
        )

    assert upstream.authorizations == [
        "Bearer sk_test_rotated_one",
        "Bearer sk_test_rotated_two",
    ]
