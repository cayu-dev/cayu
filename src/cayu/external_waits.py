"""Composable external-event admission and durable outcome election.

The host authenticates callers; a registered policy authorizes each operation.
This component never submits an external job or dispatches an agent.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, TypeVar

from cayu._validation import require_durable_clean_nonblank, revalidate_model_input
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.participants import CollaborationCapacityExceeded, CollaborationUnavailable
from cayu.sessions._external_wait_transition import ExternalWaitMutation
from cayu.sessions.base import SessionStore
from cayu.sessions.external_waits import (
    EXTERNAL_WAIT_PAGE_LIMIT,
    EXTERNAL_WAIT_PAGE_SIZE,
    ExternalCorrelation,
    ExternalCorrelationRequest,
    ExternalDeliveryReceipt,
    ExternalEventDelivery,
    ExternalWaitCapacityExceeded,
    ExternalWaitConflict,
    ExternalWaitLimits,
    ExternalWaitOutcome,
    ExternalWaitPruneResult,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitRetirement,
    ExternalWaitRetirementRequest,
    ExternalWaitScope,
    ExternalWaitUnavailable,
    Identifier,
    _Value,
    canonical_payload,
    external_wait_digest,
)
from cayu.vaults.redaction import SecretRedactor

ExternalWaitAction = Literal[
    "reserve", "register", "deliver", "read", "cancel", "service", "retire", "cleanup"
]


class ExternalWaitContext(_Value):
    """Principal supplied by the trusted SDK host, never decoded from a webhook."""

    principal: Identifier


class ExternalWaitAccessPolicy(ABC):
    @abstractmethod
    def authorize(
        self,
        context: ExternalWaitContext,
        *,
        scope: ExternalWaitScope,
        source: str,
        action: ExternalWaitAction,
    ) -> bool:
        """Return exactly True to grant this operation; perform no external effects."""


class ExternalWaitProjector(ABC):
    @abstractmethod
    def project(self, outcome: ExternalWaitOutcome) -> str:
        """Return deterministic bounded JSON; no side effects or execution authority."""


class JsonExternalWaitProjector(ExternalWaitProjector):
    def project(self, outcome: ExternalWaitOutcome) -> str:
        # Outcome identity/kind is carried separately by the typed receipt.
        # Wrapping maximum-sized event JSON would exceed an equal projection
        # budget after the event had already been durably accepted.
        if outcome.kind == "event":
            assert outcome.payload_json is not None
            return outcome.payload_json
        return json.dumps({"kind": outcome.kind}, separators=(",", ":"))


class ExternalWaitSnapshot(_Value):
    correlation: ExternalCorrelation
    registration: ExternalWaitRegistration | None
    revision: int
    outcome: ExternalWaitOutcome | None
    handoff: Literal["unbound", "pending", "settled", "excluded"]
    pending_handoff: bool
    execution_excluded: bool
    retirement_requested: bool
    timer_published: bool
    pending_timer: bool
    projection_json: str | None = None


V = TypeVar("V", bound=_Value)


def _snapshot(value: V, cls: type[V]) -> V:
    # Revalidate raw model fields, not model_dump: malformed values must not
    # reach Pydantic serializer warnings before validation/redaction.
    result = None
    try:
        if type(value) is cls:
            result = revalidate_model_input(value, cls)
    except (ValueError, TypeError, AttributeError, RecursionError):
        pass
    if result is None:
        raise ValueError("Invalid external wait input.")
    return result


def _view(record: ExternalWaitRecord) -> ExternalWaitSnapshot:
    return ExternalWaitSnapshot(
        correlation=record.correlation,
        registration=record.registration,
        revision=record.revision,
        outcome=record.outcome,
        handoff=record.handoff,
        pending_handoff=record.pending_handoff,
        execution_excluded=record.execution_excluded,
        retirement_requested=record.execution_retirement is not None,
        timer_published=record.timer_published,
        pending_timer=record.timer is not None and not record.timer_published,
        projection_json=record.projection_json,
    )


class ExternalEventWaits:
    def __init__(
        self,
        *,
        store: SessionStore,
        access_policy: ExternalWaitAccessPolicy,
        limits: ExternalWaitLimits | None = None,
        redactor: SecretRedactor | None = None,
        projectors: Mapping[tuple[str, int], ExternalWaitProjector] | None = None,
    ) -> None:
        if store.supports_external_waits() is not True:
            raise NotImplementedError("Session store does not qualify external waits.")
        if not isinstance(access_policy, ExternalWaitAccessPolicy):
            raise TypeError("External waits require a registered access policy.")
        self.store = store
        self.access_policy = access_policy
        self.limits = _snapshot(limits or ExternalWaitLimits(), ExternalWaitLimits)
        self.redactor = redactor or SecretRedactor()
        self._owners = _MutationOwners()
        registered = (
            dict(projectors)
            if projectors is not None
            else {("json", 1): JsonExternalWaitProjector()}
        )
        for key, projector in registered.items():
            if type(key) is not tuple or len(key) != 2:
                raise ValueError("External projector registration requires an ID and version.")
            name, version = key
            if (
                type(name) is not str
                or len(name) > 256
                or type(version) is not int
                or not 1 <= version <= 9007199254740991
            ):
                raise ValueError("External projector registration identity is invalid.")
            require_durable_clean_nonblank(name, "external projector ID")
            if not isinstance(projector, ExternalWaitProjector):
                raise TypeError("External projector must implement its registered contract.")
        self._projectors = MappingProxyType(registered)

    def _authorize(
        self,
        request: ExternalCorrelationRequest,
        context: ExternalWaitContext,
        action: ExternalWaitAction,
    ) -> None:
        context = _snapshot(context, ExternalWaitContext)
        safe = self.redactor.redact_json(request.model_dump(mode="json"))
        if safe != request.model_dump(mode="json"):
            raise ValueError("External correlation identities must not contain secrets.")
        granted = False
        with suppress(Exception):
            granted = (
                self.access_policy.authorize(
                    context, scope=request.scope, source=request.source, action=action
                )
                is True
            )
        if not granted:
            raise PermissionError("External wait operation is not authorized.")

    async def _mutate(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        identity = command.model_dump(mode="json", exclude={"delivery": {"payload_json"}})
        if self.redactor.redact_json(identity) != identity:
            raise ValueError("External wait identities must not contain secrets.")
        digest = external_wait_digest(command)
        return await self._observe_operation(
            lambda: self.store._mutate_external_wait(command),
            key=("external-wait", digest),
            expectation=digest.encode(),
        )

    async def _observe_operation(self, operation, *, key, expectation):
        from cayu.runtime._external_wait_observation import external_wait_tracker

        try:
            return await self._owners.run(
                operation,
                key=key,
                expectation=expectation,
                redactor=self.redactor,
                track=external_wait_tracker(),
            )
        except CollaborationCapacityExceeded as error:
            raise ExternalWaitCapacityExceeded(
                "External wait observations are at capacity."
            ) from error
        except CollaborationUnavailable as error:
            raise ExternalWaitUnavailable(
                "External wait acknowledgement is unavailable; reconcile the exact operation."
            ) from error

    def _command(self, kind, correlation: ExternalCorrelation, **fields) -> ExternalWaitMutation:
        return ExternalWaitMutation(
            kind=kind,
            request=correlation.request,
            expected=correlation,
            limits=self.limits,
            **fields,
        )

    async def reserve_correlation(
        self, request: ExternalCorrelationRequest, *, context: ExternalWaitContext
    ) -> ExternalCorrelation:
        request = _snapshot(request, ExternalCorrelationRequest)
        self._authorize(request, context, "reserve")
        record = await self._mutate(
            ExternalWaitMutation(kind="reserve", request=request, limits=self.limits)
        )
        return record.correlation

    async def register(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        registration = _snapshot(registration, ExternalWaitRegistration)
        self._authorize(registration.correlation.request, context, "register")
        return _view(
            await self._mutate(
                self._command("register", registration.correlation, registration=registration)
            )
        )

    async def deliver(
        self, delivery: ExternalEventDelivery, *, context: ExternalWaitContext
    ) -> ExternalDeliveryReceipt:
        delivery = _snapshot(delivery, ExternalEventDelivery)
        self._authorize(delivery.correlation.request, context, "deliver")
        # Commit the original validated content identity before redaction. Two
        # different secrets redacted to the same marker are not exact replay.
        content_sha256 = sha256(delivery.payload_json.encode()).hexdigest()
        payload = self.redactor.redact_json(json.loads(delivery.payload_json))
        sanitized = canonical_payload(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            self.limits.payload_bytes,
        )
        delivery = delivery.model_copy(update={"payload_json": sanitized})
        result = await self._mutate(
            self._command(
                "deliver", delivery.correlation, delivery=delivery, content_sha256=content_sha256
            )
        )
        return next(
            receipt for receipt in result.deliveries if receipt.delivery_id == delivery.delivery_id
        )

    async def project(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        """Freeze result input once; replay never reruns a previously committed projector."""
        registration = _snapshot(registration, ExternalWaitRegistration)
        self._authorize(registration.correlation.request, context, "service")
        current = await self.store._read_external_wait(
            registration.correlation.request.scope,
            registration.correlation.request.correlation_key,
        )
        if current is None or current.registration != registration:
            raise ExternalWaitConflict(
                "External projection registration is unavailable or conflicts."
            )
        if current.projection_json is not None:
            return _view(current)
        if current.outcome is None or current.outcome.kind not in {"event", "timeout"}:
            raise ExternalWaitUnavailable("External wait has no outcome eligible for continuation.")
        projector = self._projectors.get(
            (registration.projector_id, registration.projector_version)
        )
        if projector is None:
            raise ExternalWaitUnavailable("The registered external projector is unavailable.")
        projection = None
        # Opaque projector failures and malformed return values must not disclose
        # private diagnostics. No native mutation has been entered at this point.
        with suppress(Exception):
            raw = projector.project(current.outcome.model_copy(deep=True))
            canonical = canonical_payload(raw, current.correlation.limits.projection_bytes)
            redacted = self.redactor.redact_json(json.loads(canonical))
            projection = canonical_payload(
                json.dumps(redacted, ensure_ascii=False, separators=(",", ":")),
                current.correlation.limits.projection_bytes,
            )
        if projection is None:
            raise ExternalWaitUnavailable("External result projection failed before publication.")
        return _view(
            await self._mutate(
                self._command(
                    "project",
                    registration.correlation,
                    registration=registration,
                    expected_outcome=current.outcome,
                    projection_json=projection,
                )
            )
        )

    async def cancel(
        self, correlation: ExternalCorrelation, *, operation_key: str, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        correlation = _snapshot(correlation, ExternalCorrelation)
        self._authorize(correlation.request, context, "cancel")
        return _view(
            await self._mutate(self._command("cancel", correlation, operation_key=operation_key))
        )

    async def observe(
        self, correlation: ExternalCorrelation, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        correlation = _snapshot(correlation, ExternalCorrelation)
        self._authorize(correlation.request, context, "service")
        return _view(await self._mutate(self._command("observe", correlation)))

    async def lookup(
        self, request: ExternalCorrelationRequest, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot | None:
        request = _snapshot(request, ExternalCorrelationRequest)
        self._authorize(request, context, "read")
        record = await self.store._read_external_wait(request.scope, request.correlation_key)
        if record is not None and record.correlation.request != request:
            raise ExternalWaitConflict("External correlation lookup intent conflicts.")
        return None if record is None else _view(record)

    async def inspect(
        self, correlation: ExternalCorrelation, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        correlation = _snapshot(correlation, ExternalCorrelation)
        result = await self.lookup(correlation.request, context=context)
        if result is None or result.correlation != correlation:
            raise ExternalWaitConflict("External correlation incarnation is unavailable.")
        return result

    async def list(
        self,
        *,
        scope: ExternalWaitScope,
        source: str,
        context: ExternalWaitContext,
        after: str = "",
        limit: int = EXTERNAL_WAIT_PAGE_SIZE,
    ) -> tuple[ExternalWaitSnapshot, ...]:
        scope = _snapshot(scope, ExternalWaitScope)
        if type(limit) is not int or not 1 <= limit <= EXTERNAL_WAIT_PAGE_LIMIT:
            raise ValueError("External wait page limit is invalid.")
        if type(after) is not str or len(after) > 256:
            raise ValueError("External wait cursor is invalid.")
        self._authorize(
            ExternalCorrelationRequest(scope=scope, source=source, correlation_key="discovery"),
            context,
            "read",
        )
        rows = await self.store._list_external_waits(scope, source=source, after=after, limit=limit)
        for record in rows:
            self._authorize(record.correlation.request, context, "read")
        return tuple(_view(record) for record in rows)

    async def retire_scope(
        self, scope: ExternalWaitScope, *, operation_key: str, context: ExternalWaitContext
    ) -> ExternalWaitRetirement:
        """Fence a terminal generation permanently; requires scope-wide authority.

        The policy receives source="*" and action="retire". Unresolved runtime
        or timer handoffs reject retirement without changing any records.
        """
        request = ExternalWaitRetirementRequest(
            scope=_snapshot(scope, ExternalWaitScope),
            operation_key=operation_key,
            limits=self.limits,
        )
        self._authorize(
            ExternalCorrelationRequest(
                scope=request.scope, source="*", correlation_key=operation_key
            ),
            context,
            "retire",
        )
        digest = external_wait_digest(request)
        return await self._observe_operation(
            lambda: self.store._retire_external_wait_scope(request),
            key=("external-wait-retire", digest),
            expectation=digest.encode(),
        )

    async def prune_retired_scope(
        self,
        retirement: ExternalWaitRetirement,
        *,
        context: ExternalWaitContext,
        limit: int = EXTERNAL_WAIT_PAGE_SIZE,
    ) -> ExternalWaitPruneResult:
        """Remove a bounded terminal page, retaining the exact retirement tombstone.

        Retrying after acknowledgement loss may prune the next page. This is
        maintenance progress, not a replayable per-record deletion receipt.
        """
        from cayu.sessions._external_wait_retirement import require_prune_limit

        require_prune_limit(limit)
        retirement = _snapshot(retirement, ExternalWaitRetirement)
        if retirement.request.limits != self.limits:
            raise ExternalWaitConflict("External wait scope limits conflict.")
        self._authorize(
            ExternalCorrelationRequest(
                scope=retirement.request.scope,
                source="*",
                correlation_key=retirement.request.operation_key,
            ),
            context,
            "cleanup",
        )
        digest = external_wait_digest(retirement)
        return await self._observe_operation(
            lambda: self.store._prune_external_wait_scope(retirement, limit=limit),
            key=("external-wait-prune", digest, limit),
            expectation=f"{digest}:{limit}".encode(),
        )

    async def observe_page(
        self,
        *,
        scope: ExternalWaitScope,
        source: str,
        context: ExternalWaitContext,
        after: str = "",
        limit: int = EXTERNAL_WAIT_PAGE_SIZE,
    ) -> tuple[ExternalWaitSnapshot, ...]:
        """Elect due outcomes in a bounded page using the native owner clock.

        Each item commits separately. Resume from the last returned correlation
        key; interrupted pages are safe to repeat. No worker starts implicitly.
        """
        page = await self.list(
            scope=scope, source=source, context=context, after=after, limit=limit
        )
        results = []
        for item in page:
            results.append(await self.observe(item.correlation, context=context))
        return tuple(results)

    async def aclose(self, *, timeout_s: float = 10.0) -> bool:
        await self._owners.drain(timeout_s=timeout_s)
        return not self._owners.outstanding()
