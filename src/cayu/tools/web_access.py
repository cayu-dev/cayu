"""Typed web-access evidence and explicit durable WebBridge routing."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.tools import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _runtime_tool_invocation_authority,
    _RuntimeToolInvocationAuthority,
)

if TYPE_CHECKING:
    from cayu.tools.webbridge import WebBridge

MAX_WEB_ACCESS_ROUTES = 16
MAX_WEB_ACCESS_RULES = 128
MAX_WEB_ACCESS_CIRCUIT_ENTRIES = 128
MAX_WEB_ACCESS_ROUTE_ID_BYTES = 64
MAX_WEB_ACCESS_GUIDANCE_BYTES = 512
MAX_WEB_ACCESS_RETRY_AFTER_SECONDS = 24 * 60 * 60
_ROUTE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CIRCUIT_RECORD_TYPE = "cayu.web-access-circuit"
_CIRCUIT_SCHEMA_VERSION = 1
_MAX_CAS_ATTEMPTS = 4
_CIRCUIT_ENTRY_FIELDS = frozenset(
    {
        "authority_sha256",
        "fingerprint",
        "route_id",
        "request_destination_fingerprint",
        "destination_fingerprint",
        "effective_source_url",
        "outcome",
        "source",
        "signal",
        "status_code",
        "retry_after_seconds",
        "retry_after_unrepresentable",
        "denial_count",
        "next_eligible_at",
        "updated_at",
    }
)


class WebAccessOutcome(StrEnum):
    """Stable access barriers derived from trusted transport observations."""

    AUTHENTICATION_REQUIRED = "authentication_required"
    CONSENT_REQUIRED = "consent_required"
    RATE_LIMITED = "rate_limited"
    BOT_CHALLENGE = "bot_challenge"
    DESTINATION_DENIED = "destination_denied"
    CONTENT_UNAVAILABLE = "content_unavailable"
    TRANSIENT_TRANSPORT_FAILURE = "transient_transport_failure"


class WebAccessEvidenceSource(StrEnum):
    HTTP_RESPONSE = "http_response"
    BROWSER_RESPONSE = "browser_response"
    EGRESS_POLICY = "egress_policy"
    HOSTED_PROVIDER = "hosted_provider"
    TRANSPORT = "transport"


class WebAccessSignal(StrEnum):
    STATUS_CODE = "status_code"
    WWW_AUTHENTICATE = "www_authenticate"
    RETRY_AFTER = "retry_after"
    CHALLENGE_HEADER = "challenge_header"
    CONSENT_HEADER = "consent_header"
    EGRESS_DENIAL = "egress_denial"
    TRANSPORT_ERROR = "transport_error"
    PROVIDER_STATUS = "provider_status"


_BUILTIN_ADAPTER_ACCESS_SOURCES: dict[type[object], WebAccessEvidenceSource] = {}


def _register_builtin_adapter_access_source(
    adapter_type: type[object],
    source: WebAccessEvidenceSource,
) -> None:
    if type(adapter_type) is not type or not isinstance(source, WebAccessEvidenceSource):
        raise TypeError("Built-in web adapter access authority is invalid.")
    existing = _BUILTIN_ADAPTER_ACCESS_SOURCES.get(adapter_type)
    if existing is not None and existing is not source:
        raise RuntimeError("Built-in web adapter access authority conflicts.")
    _BUILTIN_ADAPTER_ACCESS_SOURCES[adapter_type] = source


def _builtin_adapter_access_source(adapter: object) -> WebAccessEvidenceSource:
    return _BUILTIN_ADAPTER_ACCESS_SOURCES.get(
        type(adapter),
        WebAccessEvidenceSource.HOSTED_PROVIDER,
    )


class WebAccessEvidence(BaseModel):
    """Content-free, bounded evidence for one classified access barrier."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    outcome: WebAccessOutcome
    source: WebAccessEvidenceSource
    signal: WebAccessSignal
    destination_fingerprint: str = Field(min_length=64, max_length=64)
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    retry_after_seconds: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_WEB_ACCESS_RETRY_AFTER_SECONDS,
    )
    retry_after_unrepresentable: StrictBool = False

    @model_validator(mode="after")
    def validate_evidence(self) -> WebAccessEvidence:
        if _SHA256.fullmatch(self.destination_fingerprint) is None:
            raise ValueError("destination_fingerprint must be lowercase SHA-256 hex.")
        if self.outcome is WebAccessOutcome.RATE_LIMITED:
            if self.status_code not in {None, 429}:
                raise ValueError("Rate-limit evidence may carry only HTTP status 429.")
            if self.retry_after_unrepresentable and self.retry_after_seconds is not None:
                raise ValueError("Unrepresentable retry timing cannot carry a bounded delay.")
            if self.retry_after_unrepresentable and self.signal is not WebAccessSignal.RETRY_AFTER:
                raise ValueError("Unrepresentable retry timing requires Retry-After evidence.")
        elif self.retry_after_seconds is not None:
            raise ValueError("Only rate-limit evidence may carry retry_after_seconds.")
        elif self.retry_after_unrepresentable:
            raise ValueError("Only rate-limit evidence may carry retry timing authority.")
        if self.source is WebAccessEvidenceSource.EGRESS_POLICY:
            if (
                self.outcome is not WebAccessOutcome.DESTINATION_DENIED
                or self.signal is not WebAccessSignal.EGRESS_DENIAL
                or self.status_code is not None
            ):
                raise ValueError("Egress evidence must be a content-free destination denial.")
        elif self.source is WebAccessEvidenceSource.TRANSPORT:
            if (
                self.outcome is not WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE
                or self.signal is not WebAccessSignal.TRANSPORT_ERROR
                or self.status_code is not None
            ):
                raise ValueError("Transport evidence must be a content-free transport failure.")
        elif self.source in {
            WebAccessEvidenceSource.HTTP_RESPONSE,
            WebAccessEvidenceSource.BROWSER_RESPONSE,
        }:
            if self.status_code is None or self.signal in {
                WebAccessSignal.EGRESS_DENIAL,
                WebAccessSignal.PROVIDER_STATUS,
                WebAccessSignal.TRANSPORT_ERROR,
            }:
                raise ValueError("Response evidence requires bounded response metadata.")
        elif self.signal not in {
            WebAccessSignal.PROVIDER_STATUS,
            WebAccessSignal.RETRY_AFTER,
            WebAccessSignal.STATUS_CODE,
        }:
            raise ValueError("Hosted-provider evidence uses only provider or status signals.")
        expected_outcome = {
            WebAccessSignal.WWW_AUTHENTICATE: WebAccessOutcome.AUTHENTICATION_REQUIRED,
            WebAccessSignal.RETRY_AFTER: WebAccessOutcome.RATE_LIMITED,
            WebAccessSignal.CHALLENGE_HEADER: WebAccessOutcome.BOT_CHALLENGE,
            WebAccessSignal.CONSENT_HEADER: WebAccessOutcome.CONSENT_REQUIRED,
            WebAccessSignal.EGRESS_DENIAL: WebAccessOutcome.DESTINATION_DENIED,
            WebAccessSignal.TRANSPORT_ERROR: WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE,
        }.get(self.signal)
        if expected_outcome is not None and self.outcome is not expected_outcome:
            raise ValueError("Access outcome conflicts with its trusted signal.")
        if self.signal is WebAccessSignal.STATUS_CODE and not _status_supports_outcome(
            self.status_code,
            self.outcome,
        ):
            raise ValueError("Access outcome conflicts with its response status.")
        return self


class _CircuitEntry(TypedDict):
    authority_sha256: str
    fingerprint: str
    route_id: str
    request_destination_fingerprint: str
    destination_fingerprint: str
    effective_source_url: str | None
    outcome: str
    source: str
    signal: str
    status_code: int | None
    retry_after_seconds: int | None
    retry_after_unrepresentable: bool
    denial_count: int
    next_eligible_at: int | None
    updated_at: int


class WebAccessRouteActionKind(StrEnum):
    FALLBACK = "fallback"
    WAIT = "wait"
    OPERATOR_ACTION = "operator_action"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class WebAccessRouteAction:
    """One application-owned action for one classified route outcome."""

    kind: WebAccessRouteActionKind
    target_route_id: str | None = None
    wait_seconds: int | None = None
    guidance: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WebAccessRouteActionKind):
            raise TypeError("kind must be a WebAccessRouteActionKind.")
        if self.kind is WebAccessRouteActionKind.FALLBACK:
            if self.target_route_id is None:
                raise ValueError("Fallback actions require target_route_id.")
            _route_id(self.target_route_id)
        elif self.target_route_id is not None:
            raise ValueError("Only fallback actions may declare target_route_id.")
        if self.kind is WebAccessRouteActionKind.WAIT:
            if (
                type(self.wait_seconds) is not int
                or self.wait_seconds < 1
                or self.wait_seconds > MAX_WEB_ACCESS_RETRY_AFTER_SECONDS
            ):
                raise ValueError("Wait actions require wait_seconds between 1 and 86400.")
        elif self.wait_seconds is not None:
            raise ValueError("Only wait actions may declare wait_seconds.")
        if self.guidance is not None:
            guidance = require_durable_text(self.guidance, "guidance")
            if len(guidance.encode("utf-8")) > MAX_WEB_ACCESS_GUIDANCE_BYTES:
                raise ValueError("guidance must not exceed 512 bytes.")
            object.__setattr__(self, "guidance", guidance)

    @classmethod
    def fallback_to(cls, route_id: str) -> WebAccessRouteAction:
        return cls(WebAccessRouteActionKind.FALLBACK, target_route_id=route_id)

    @classmethod
    def wait(cls, seconds: int, *, guidance: str | None = None) -> WebAccessRouteAction:
        return cls(WebAccessRouteActionKind.WAIT, wait_seconds=seconds, guidance=guidance)

    @classmethod
    def operator_action(cls, guidance: str) -> WebAccessRouteAction:
        return cls(WebAccessRouteActionKind.OPERATOR_ACTION, guidance=guidance)

    @classmethod
    def stop(cls, *, guidance: str | None = None) -> WebAccessRouteAction:
        return cls(WebAccessRouteActionKind.STOP, guidance=guidance)


@dataclass(frozen=True, slots=True)
class WebAccessRouteRule:
    route_id: str
    outcome: WebAccessOutcome
    action: WebAccessRouteAction

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_id", _route_id(self.route_id))
        if not isinstance(self.outcome, WebAccessOutcome):
            raise TypeError("outcome must be a WebAccessOutcome.")
        if type(self.action) is not WebAccessRouteAction:
            raise TypeError("action must be a WebAccessRouteAction.")


@dataclass(frozen=True, slots=True)
class WebAccessCircuitPolicy:
    threshold: int = 3
    open_seconds: int = 300
    max_entries: int = 64

    def __post_init__(self) -> None:
        if type(self.threshold) is not int or not 1 <= self.threshold <= 32:
            raise ValueError("threshold must be an integer between 1 and 32.")
        if (
            type(self.open_seconds) is not int
            or not 1 <= self.open_seconds <= MAX_WEB_ACCESS_RETRY_AFTER_SECONDS
        ):
            raise ValueError("open_seconds must be an integer between 1 and 86400.")
        if (
            type(self.max_entries) is not int
            or not 1 <= self.max_entries <= MAX_WEB_ACCESS_CIRCUIT_ENTRIES
        ):
            raise ValueError("max_entries must be an integer between 1 and 128.")


@dataclass(frozen=True, slots=True)
class WebAccessRoutePolicy:
    entry_route_id: str
    rules: tuple[WebAccessRouteRule, ...] = ()
    circuit: WebAccessCircuitPolicy = WebAccessCircuitPolicy()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_route_id", _route_id(self.entry_route_id))
        if type(self.rules) is not tuple or len(self.rules) > MAX_WEB_ACCESS_RULES:
            raise ValueError("rules must be a tuple with at most 128 entries.")
        if any(type(rule) is not WebAccessRouteRule for rule in self.rules):
            raise TypeError("rules must contain WebAccessRouteRule values.")
        keys = [(rule.route_id, rule.outcome) for rule in self.rules]
        if len(set(keys)) != len(keys):
            raise ValueError("rules must not repeat a route/outcome pair.")
        if type(self.circuit) is not WebAccessCircuitPolicy:
            raise TypeError("circuit must be a WebAccessCircuitPolicy.")


@dataclass(frozen=True, slots=True)
class WebBridgeRoute:
    """One explicitly named already-validated WebBridge fetch route."""

    route_id: str
    bridge: WebBridge

    def __post_init__(self) -> None:
        from cayu.tools.webbridge import WebBridge, WebBridgeProfileKind

        object.__setattr__(self, "route_id", _route_id(self.route_id))
        if type(self.bridge) is not WebBridge:
            raise TypeError("bridge must be a validated WebBridge.")
        if self.bridge.kind is WebBridgeProfileKind.ROUTED:
            raise ValueError("Routed WebBridges cannot contain another routed bridge.")
        fetch_tools = tuple(
            tool for tool in self.bridge.tools if getattr(tool, "name", None) == "web_fetch"
        )
        if len(fetch_tools) != 1:
            raise ValueError("Every WebBridge route must expose exactly one web_fetch tool.")


class WebAccessRoutingTool(Tool):
    """Execute only the finite application-owned WebBridge route graph."""

    spec = ToolSpec(
        name="web_fetch",
        effect=ToolEffect.NONE,
        parallel_safe=False,
        description=(
            "Fetch a public HTTPS page through an explicit application-owned route policy. "
            "Access blocks and replacement evidence remain bounded and attributable."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "minLength": 1,
                    "maxLength": 8192,
                }
            },
            "required": ["url"],
        },
    )

    def __init__(
        self,
        *,
        routes: Sequence[WebBridgeRoute],
        policy: WebAccessRoutePolicy,
    ) -> None:
        if isinstance(routes, (str, bytes)) or not isinstance(routes, Sequence):
            raise TypeError("routes must be a sequence of WebBridgeRoute values.")
        owned_routes = tuple(routes)
        if not 1 <= len(owned_routes) <= MAX_WEB_ACCESS_ROUTES:
            raise ValueError("routes must contain between 1 and 16 entries.")
        if any(type(route) is not WebBridgeRoute for route in owned_routes):
            raise TypeError("routes must contain WebBridgeRoute values.")
        route_ids = tuple(route.route_id for route in owned_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("route ids must be unique.")
        if type(policy) is not WebAccessRoutePolicy:
            raise TypeError("policy must be a WebAccessRoutePolicy.")
        if policy.entry_route_id not in route_ids:
            raise ValueError("entry_route_id must identify a configured route.")
        for rule in policy.rules:
            if rule.route_id not in route_ids:
                raise ValueError("Every rule must identify a configured route.")
            target = rule.action.target_route_id
            if target is not None and target not in route_ids:
                raise ValueError("Every fallback target must identify a configured route.")
            if target == rule.route_id:
                raise ValueError("A route cannot fall back to itself.")
        _require_acyclic_fallbacks(policy.rules)
        self.routes = owned_routes
        self.policy = policy
        self._routes = {route.route_id: route for route in owned_routes}
        self._rules = {(rule.route_id, rule.outcome): rule.action for rule in policy.rules}
        self._route_identities = {
            route.route_id: _bridge_route_identity(route) for route in owned_routes
        }
        self.policy_fingerprint = _policy_fingerprint(
            policy,
            self._route_identities,
        )
        super().__init__(self.spec)

    def _execution_profile_material(self) -> dict[str, Any] | None:
        route_material: list[dict[str, Any]] = []
        for route in self.routes:
            material = _route_fetch_material(route)
            if material is None:
                return None
            route_material.append(
                {
                    "route_id": route.route_id,
                    "profile_fingerprint": self._route_identities[route.route_id][
                        "profile_fingerprint"
                    ],
                    "tool": material,
                }
            )
        return {
            "component": "cayu.tools.web_access:WebAccessRoutingTool",
            "policy_fingerprint": self.policy_fingerprint,
            "routes": route_material,
        }

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        requested_url = _requested_url(args)
        if requested_url is None:
            return _plain_error("invalid_url", "A valid public HTTPS URL is required.")
        authority = _routing_authority(ctx, args)
        if authority is None:
            return _plain_error(
                "durable_authority_unavailable",
                "Web access routing requires durable runtime invocation authority.",
            )
        execution_profile_fingerprint = authority.execution_profile_fingerprint
        now = _utc_now_seconds()
        history: list[dict[str, Any]] = []
        original: WebAccessEvidence | None = None
        pending_effective_origins: dict[str, str] = {}
        route_id = self.policy.entry_route_id
        visited: set[str] = set()
        while len(history) < len(self.routes):
            if route_id in visited:
                return _routing_terminal(
                    result=_plain_error(
                        "route_conflict", "The web route graph could not converge."
                    ),
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition="route_conflict",
                )
            visited.add(route_id)
            route = self._routes[route_id]
            try:
                open_evidence = await self._open_circuit_evidence(
                    authority,
                    requested_url=requested_url,
                    route_id=route_id,
                    now=now,
                )
            except Exception:
                return _routing_terminal(
                    result=_plain_error(
                        "durable_authority_unavailable",
                        "Durable web-access routing authority is unavailable.",
                    ),
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition="durable_authority_unavailable",
                )
            if open_evidence is not None:
                evidence, next_eligible_at, effective_source_url = open_evidence
                if original is None:
                    original = evidence
                action = self._rules.get((route_id, evidence.outcome))
                history.append(
                    _history_entry(
                        route_identity=self._route_identities[route_id],
                        evidence=evidence,
                        action=action,
                        invoked=False,
                        next_eligible_at=next_eligible_at,
                    )
                )
                decision = _route_decision(
                    action,
                    evidence=evidence,
                    circuit_next_eligible_at=next_eligible_at,
                    now=now,
                )
                if decision[0] == "fallback":
                    route_id = decision[1]
                    continue
                try:
                    await self._finalize_pending_circuit_origins(
                        authority,
                        pending_effective_origins,
                    )
                except Exception:
                    return _routing_terminal(
                        result=_plain_error(
                            "durable_authority_unavailable",
                            "Durable web-access routing authority is unavailable.",
                        ),
                        history=history,
                        original=original,
                        selected=self._route_identities[route_id],
                        execution_profile_fingerprint=execution_profile_fingerprint,
                        disposition="durable_authority_unavailable",
                    )
                return _access_terminal_result(
                    evidence=evidence,
                    action=action,
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition=decision[0],
                    next_eligible_at=decision[2],
                    effective_source_url=effective_source_url,
                )

            route_context = _route_invocation_context(
                ctx,
                policy_fingerprint=self.policy_fingerprint,
                route_id=route_id,
            )
            try:
                result = await _route_fetch_tool(route).run(route_context, dict(args))
                if type(result) is not ToolResult:
                    raise TypeError("A WebBridge route returned an invalid ToolResult.")
            except Exception:
                history.append(
                    {
                        "route": dict(self._route_identities[route_id]),
                        "invoked": True,
                        "disposition": "route_failed",
                    }
                )
                try:
                    await self._finalize_pending_circuit_origins(
                        authority,
                        pending_effective_origins,
                    )
                except Exception:
                    history[-1]["disposition"] = "failure_unrecorded"
                    return _routing_terminal(
                        result=_plain_error(
                            "durable_authority_unavailable",
                            "Durable web-access routing authority is unavailable.",
                        ),
                        history=history,
                        original=original,
                        selected=self._route_identities[route_id],
                        execution_profile_fingerprint=execution_profile_fingerprint,
                        disposition="durable_authority_unavailable",
                        default_effective_source_url=requested_url,
                    )
                return _routing_terminal(
                    result=_plain_error(
                        "route_failed",
                        "The selected WebBridge route failed.",
                    ),
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition="route_failed",
                    default_effective_source_url=requested_url,
                )
            now = _utc_now_seconds()
            access_source, allowed_sources = _route_access_sources(route)
            evidence = access_evidence_from_result(
                result,
                requested_url=requested_url,
                source=access_source,
                allowed_sources=allowed_sources,
            )
            if evidence is None:
                if not result.is_error:
                    successful_source_url = _result_effective_source_url(
                        result,
                        default=requested_url,
                    )
                    try:
                        await self._record_success(
                            authority,
                            requested_url=requested_url,
                            route_id=route_id,
                            pending_effective_origins=pending_effective_origins,
                        )
                    except Exception:
                        history.append(
                            {
                                "route": dict(self._route_identities[route_id]),
                                "invoked": True,
                                "disposition": "success_unrecorded",
                            }
                        )
                        return _routing_terminal(
                            result=_plain_error(
                                "durable_authority_unavailable",
                                "Durable web-access routing authority is unavailable.",
                            ),
                            history=history,
                            original=original,
                            selected=self._route_identities[route_id],
                            execution_profile_fingerprint=execution_profile_fingerprint,
                            disposition="durable_authority_unavailable",
                            effective_source_url=successful_source_url,
                        )
                if result.is_error and pending_effective_origins:
                    try:
                        await self._finalize_pending_circuit_origins(
                            authority,
                            pending_effective_origins,
                        )
                    except Exception:
                        history.append(
                            {
                                "route": dict(self._route_identities[route_id]),
                                "invoked": True,
                                "disposition": "failure_unrecorded",
                            }
                        )
                        return _routing_terminal(
                            result=_plain_error(
                                "durable_authority_unavailable",
                                "Durable web-access routing authority is unavailable.",
                            ),
                            history=history,
                            original=original,
                            selected=self._route_identities[route_id],
                            execution_profile_fingerprint=execution_profile_fingerprint,
                            disposition="durable_authority_unavailable",
                        )
                disposition = "success" if not result.is_error else "route_failed"
                if original is not None and not result.is_error:
                    disposition = "fallback_succeeded"
                history.append(
                    {
                        "route": dict(self._route_identities[route_id]),
                        "invoked": True,
                        "disposition": disposition,
                    }
                )
                return _routing_terminal(
                    result=result,
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition=disposition,
                    default_effective_source_url=requested_url,
                )
            if original is None:
                original = evidence
            action = self._rules.get((route_id, evidence.outcome))
            effective_source_url = _result_effective_source_origin(result)
            if effective_source_url is None:
                effective_source_url = _safe_effective_origin(requested_url)
            try:
                next_eligible_at = await self._record_denial(
                    authority,
                    evidence=evidence,
                    route_id=route_id,
                    requested_url=requested_url,
                    effective_source_url=effective_source_url,
                    action=action,
                    now=now,
                    pending_effective_origins=pending_effective_origins,
                    finalize=action is None or action.kind is not WebAccessRouteActionKind.FALLBACK,
                )
            except Exception:
                history.append(
                    _history_entry(
                        route_identity=self._route_identities[route_id],
                        evidence=evidence,
                        action=action,
                        invoked=True,
                        next_eligible_at=None,
                    )
                )
                return _routing_terminal(
                    result=_plain_error(
                        "durable_authority_unavailable",
                        "Durable web-access routing authority is unavailable.",
                    ),
                    history=history,
                    original=original,
                    selected=self._route_identities[route_id],
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    disposition="durable_authority_unavailable",
                )
            history.append(
                _history_entry(
                    route_identity=self._route_identities[route_id],
                    evidence=evidence,
                    action=action,
                    invoked=True,
                    next_eligible_at=next_eligible_at,
                )
            )
            decision = _route_decision(
                action,
                evidence=evidence,
                circuit_next_eligible_at=next_eligible_at,
                now=now,
            )
            if decision[0] == "fallback":
                route_id = decision[1]
                continue
            return _access_terminal_result(
                evidence=evidence,
                action=action,
                history=history,
                original=original,
                selected=self._route_identities[route_id],
                execution_profile_fingerprint=execution_profile_fingerprint,
                disposition=decision[0],
                next_eligible_at=decision[2],
                effective_source_url=effective_source_url,
            )
        return _routing_terminal(
            result=_plain_error("route_conflict", "The web route graph exhausted its routes."),
            history=history,
            original=original,
            selected=self._route_identities[route_id],
            execution_profile_fingerprint=execution_profile_fingerprint,
            disposition="route_conflict",
        )

    async def _open_circuit_evidence(
        self,
        authority: _RuntimeToolInvocationAuthority,
        *,
        requested_url: str,
        route_id: str,
        now: int,
    ) -> tuple[WebAccessEvidence, int | None, str | None] | None:
        record = await authority.load_durable_operation(self._circuit_key())
        parsed = _validated_circuit_record(
            record,
            policy_fingerprint=self.policy_fingerprint,
            route_ids=frozenset(self._routes),
            circuit=self.policy.circuit,
        )
        destination = web_destination_fingerprint(requested_url)
        open_entries = [
            entry
            for entry in parsed["entries"]
            if entry["route_id"] == route_id
            and entry["request_destination_fingerprint"] == destination
            and (
                entry["retry_after_unrepresentable"]
                or (entry["next_eligible_at"] is not None and entry["next_eligible_at"] > now)
            )
        ]
        if not open_entries:
            return None
        entry = max(
            open_entries,
            key=lambda item: (
                item["retry_after_unrepresentable"],
                item["next_eligible_at"] or 0,
                item["fingerprint"],
            ),
        )
        return (
            WebAccessEvidence(
                outcome=WebAccessOutcome(entry["outcome"]),
                source=WebAccessEvidenceSource(entry["source"]),
                signal=WebAccessSignal(entry["signal"]),
                destination_fingerprint=entry["destination_fingerprint"],
                status_code=entry["status_code"],
                retry_after_seconds=entry["retry_after_seconds"],
                retry_after_unrepresentable=entry["retry_after_unrepresentable"],
            ),
            entry["next_eligible_at"],
            entry["effective_source_url"],
        )

    async def _record_denial(
        self,
        authority: _RuntimeToolInvocationAuthority,
        *,
        evidence: WebAccessEvidence,
        route_id: str,
        requested_url: str,
        effective_source_url: str | None,
        action: WebAccessRouteAction | None,
        now: int,
        pending_effective_origins: dict[str, str],
        finalize: bool,
    ) -> int | None:
        key = self._circuit_key()
        request_destination = web_destination_fingerprint(requested_url)
        fingerprint = _denial_fingerprint(route_id, request_destination, evidence)
        if effective_source_url is not None:
            pending_effective_origins[fingerprint] = effective_source_url
        for _ in range(_MAX_CAS_ATTEMPTS):
            raw = await authority.load_durable_operation(key)
            record = _validated_circuit_record(
                raw,
                policy_fingerprint=self.policy_fingerprint,
                route_ids=frozenset(self._routes),
                circuit=self.policy.circuit,
            )
            entries = [cast("_CircuitEntry", dict(entry)) for entry in record["entries"]]
            matching = next(
                (entry for entry in entries if entry["fingerprint"] == fingerprint),
                None,
            )
            if matching is None:
                if len(entries) >= self.policy.circuit.max_entries:
                    eligible = [
                        entry
                        for entry in entries
                        if not entry["retry_after_unrepresentable"]
                        and (entry["next_eligible_at"] is None or entry["next_eligible_at"] <= now)
                    ]
                    if not eligible:
                        raise RuntimeError("The durable web-access circuit is at capacity.")
                    entries.remove(min(eligible, key=lambda item: item["updated_at"]))
                matching = _CircuitEntry(
                    authority_sha256="",
                    fingerprint=fingerprint,
                    route_id=route_id,
                    request_destination_fingerprint=request_destination,
                    destination_fingerprint=evidence.destination_fingerprint,
                    effective_source_url=effective_source_url,
                    outcome=evidence.outcome.value,
                    source=evidence.source.value,
                    signal=evidence.signal.value,
                    status_code=evidence.status_code,
                    retry_after_seconds=evidence.retry_after_seconds,
                    retry_after_unrepresentable=evidence.retry_after_unrepresentable,
                    denial_count=0,
                    next_eligible_at=None,
                    updated_at=now,
                )
                entries.append(matching)
            if matching["next_eligible_at"] is not None and matching["next_eligible_at"] <= now:
                matching["next_eligible_at"] = None
            matching["source"] = evidence.source.value
            matching["signal"] = evidence.signal.value
            matching["effective_source_url"] = effective_source_url
            matching["status_code"] = evidence.status_code
            matching["retry_after_seconds"] = evidence.retry_after_seconds
            matching["retry_after_unrepresentable"] = evidence.retry_after_unrepresentable
            matching["denial_count"] = min(
                self.policy.circuit.threshold,
                matching["denial_count"] + 1,
            )
            matching["updated_at"] = now
            if evidence.retry_after_unrepresentable:
                matching["next_eligible_at"] = None
            else:
                deadlines: list[int] = []
                if evidence.retry_after_seconds is not None:
                    deadlines.append(now + evidence.retry_after_seconds)
                if action is not None and action.kind is WebAccessRouteActionKind.WAIT:
                    deadlines.append(now + (action.wait_seconds or 0))
                if matching["denial_count"] >= self.policy.circuit.threshold:
                    deadlines.append(now + self.policy.circuit.open_seconds)
                if deadlines:
                    matching["next_eligible_at"] = max(
                        matching["next_eligible_at"] or 0,
                        *deadlines,
                    )
            desired = {
                "record_type": _CIRCUIT_RECORD_TYPE,
                "schema_version": _CIRCUIT_SCHEMA_VERSION,
                "policy_fingerprint": self.policy_fingerprint,
                "entries": sorted(entries, key=lambda item: item["fingerprint"]),
            }
            if finalize:
                _apply_pending_effective_origins(desired["entries"], pending_effective_origins)
                desired = _seal_circuit_record(
                    authority,
                    desired,
                    policy_fingerprint=self.policy_fingerprint,
                    route_ids=frozenset(self._routes),
                    circuit=self.policy.circuit,
                )
            else:
                desired = _prepare_intermediate_circuit_record(
                    desired,
                    pending_effective_origins=pending_effective_origins,
                    policy_fingerprint=self.policy_fingerprint,
                    route_ids=frozenset(self._routes),
                    circuit=self.policy.circuit,
                )
            try:
                await authority.compare_and_set_durable_operation(key, raw, desired, {})
            except Exception:
                persisted = await authority.load_durable_operation(key)
                if persisted == desired:
                    if finalize:
                        pending_effective_origins.clear()
                    return matching["next_eligible_at"]
                continue
            if finalize:
                pending_effective_origins.clear()
            return matching["next_eligible_at"]
        raise RuntimeError("The durable web-access circuit changed concurrently.")

    async def _record_success(
        self,
        authority: _RuntimeToolInvocationAuthority,
        *,
        requested_url: str,
        route_id: str,
        pending_effective_origins: dict[str, str],
    ) -> None:
        key = self._circuit_key()
        request_destination = web_destination_fingerprint(requested_url)
        for _ in range(_MAX_CAS_ATTEMPTS):
            raw = await authority.load_durable_operation(key)
            record = _validated_circuit_record(
                raw,
                policy_fingerprint=self.policy_fingerprint,
                route_ids=frozenset(self._routes),
                circuit=self.policy.circuit,
            )
            entries = [
                cast("_CircuitEntry", dict(entry))
                for entry in record["entries"]
                if not (
                    entry["route_id"] == route_id
                    and entry["request_destination_fingerprint"] == request_destination
                )
            ]
            if len(entries) == len(record["entries"]) and not pending_effective_origins:
                return
            _apply_pending_effective_origins(entries, pending_effective_origins)
            desired = _seal_circuit_record(
                authority,
                {
                    "record_type": _CIRCUIT_RECORD_TYPE,
                    "schema_version": _CIRCUIT_SCHEMA_VERSION,
                    "policy_fingerprint": self.policy_fingerprint,
                    "entries": entries,
                },
                policy_fingerprint=self.policy_fingerprint,
                route_ids=frozenset(self._routes),
                circuit=self.policy.circuit,
            )
            try:
                await authority.compare_and_set_durable_operation(key, raw, desired, {})
            except Exception:
                persisted = await authority.load_durable_operation(key)
                if persisted == desired:
                    pending_effective_origins.clear()
                    return
                continue
            pending_effective_origins.clear()
            return
        raise RuntimeError("The durable web-access circuit changed concurrently.")

    async def _finalize_pending_circuit_origins(
        self,
        authority: _RuntimeToolInvocationAuthority,
        pending_effective_origins: dict[str, str],
    ) -> None:
        if not pending_effective_origins:
            return
        key = self._circuit_key()
        for _ in range(_MAX_CAS_ATTEMPTS):
            raw = await authority.load_durable_operation(key)
            record = _validated_circuit_record(
                raw,
                policy_fingerprint=self.policy_fingerprint,
                route_ids=frozenset(self._routes),
                circuit=self.policy.circuit,
            )
            entries = [cast("_CircuitEntry", dict(entry)) for entry in record["entries"]]
            _apply_pending_effective_origins(entries, pending_effective_origins)
            desired = _seal_circuit_record(
                authority,
                {
                    "record_type": _CIRCUIT_RECORD_TYPE,
                    "schema_version": _CIRCUIT_SCHEMA_VERSION,
                    "policy_fingerprint": self.policy_fingerprint,
                    "entries": entries,
                },
                policy_fingerprint=self.policy_fingerprint,
                route_ids=frozenset(self._routes),
                circuit=self.policy.circuit,
            )
            try:
                await authority.compare_and_set_durable_operation(key, raw, desired, {})
            except Exception:
                persisted = await authority.load_durable_operation(key)
                if persisted != desired:
                    continue
            pending_effective_origins.clear()
            return
        raise RuntimeError("The durable web-access circuit changed concurrently.")

    def _circuit_key(self) -> str:
        return f"web_access_circuit:{self.policy_fingerprint}"


def classify_http_access(
    requested_url: str,
    *,
    status_code: int,
    headers: Mapping[str, str],
    source: WebAccessEvidenceSource,
    now: datetime | None = None,
) -> WebAccessEvidence | None:
    """Classify only bounded response metadata; response content is never inspected."""

    if type(status_code) is not int or not 100 <= status_code <= 599:
        raise ValueError("status_code must be an HTTP status.")
    normalized = _bounded_access_headers(headers)
    outcome: WebAccessOutcome | None = None
    signal = WebAccessSignal.STATUS_CODE
    retry_after: int | None = None
    retry_after_unrepresentable = False
    if normalized.get("x-cayu-access-requirement", "").lower() == "consent":
        outcome = WebAccessOutcome.CONSENT_REQUIRED
        signal = WebAccessSignal.CONSENT_HEADER
    elif (
        normalized.get("cf-mitigated", "").lower() == "challenge"
        or normalized.get("x-cayu-access-block", "").lower() == "bot_challenge"
    ):
        outcome = WebAccessOutcome.BOT_CHALLENGE
        signal = WebAccessSignal.CHALLENGE_HEADER
    elif status_code == 401:
        if normalized.get("www-authenticate"):
            outcome = WebAccessOutcome.AUTHENTICATION_REQUIRED
            signal = WebAccessSignal.WWW_AUTHENTICATE
        else:
            # Anubis's default challenge response is a content-free HTTP 401
            # without WWW-Authenticate. Do not inspect or trust its page body.
            outcome = WebAccessOutcome.BOT_CHALLENGE
    elif status_code == 407:
        outcome = WebAccessOutcome.AUTHENTICATION_REQUIRED
        signal = WebAccessSignal.WWW_AUTHENTICATE
    elif status_code == 429:
        outcome = WebAccessOutcome.RATE_LIMITED
        retry_after, retry_after_unrepresentable = _retry_after_seconds(
            normalized.get("retry-after"),
            now=now,
        )
        signal = (
            WebAccessSignal.RETRY_AFTER
            if retry_after is not None or retry_after_unrepresentable
            else WebAccessSignal.STATUS_CODE
        )
    elif status_code in {403, 451}:
        outcome = WebAccessOutcome.DESTINATION_DENIED
    elif status_code in {404, 410}:
        outcome = WebAccessOutcome.CONTENT_UNAVAILABLE
    elif status_code in {408, 425} or status_code >= 500:
        outcome = WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE
    if outcome is None:
        return None
    if source is WebAccessEvidenceSource.HOSTED_PROVIDER and signal not in {
        WebAccessSignal.PROVIDER_STATUS,
        WebAccessSignal.RETRY_AFTER,
        WebAccessSignal.STATUS_CODE,
    }:
        signal = WebAccessSignal.PROVIDER_STATUS
    return WebAccessEvidence(
        outcome=outcome,
        source=source,
        signal=signal,
        destination_fingerprint=web_destination_fingerprint(requested_url),
        status_code=status_code,
        retry_after_seconds=retry_after,
        retry_after_unrepresentable=retry_after_unrepresentable,
    )


def _status_supports_outcome(
    status_code: int | None,
    outcome: WebAccessOutcome,
) -> bool:
    if status_code is None:
        return False
    allowed: set[WebAccessOutcome]
    if status_code == 401:
        allowed = {
            WebAccessOutcome.AUTHENTICATION_REQUIRED,
            WebAccessOutcome.BOT_CHALLENGE,
        }
    elif status_code == 403:
        allowed = {
            WebAccessOutcome.BOT_CHALLENGE,
            WebAccessOutcome.DESTINATION_DENIED,
        }
    elif status_code == 407:
        allowed = {WebAccessOutcome.AUTHENTICATION_REQUIRED}
    elif status_code == 428:
        allowed = {WebAccessOutcome.CONSENT_REQUIRED}
    elif status_code == 429:
        allowed = {WebAccessOutcome.RATE_LIMITED}
    elif status_code in {404, 410}:
        allowed = {WebAccessOutcome.CONTENT_UNAVAILABLE}
    elif status_code == 451:
        allowed = {WebAccessOutcome.DESTINATION_DENIED}
    elif status_code in {408, 425} or status_code >= 500:
        allowed = {WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE}
    else:
        allowed = set()
    return outcome in allowed


def transport_access_evidence(
    requested_url: str,
    *,
    outcome: WebAccessOutcome,
    source: WebAccessEvidenceSource,
    signal: WebAccessSignal,
) -> WebAccessEvidence:
    return WebAccessEvidence(
        outcome=outcome,
        source=source,
        signal=signal,
        destination_fingerprint=web_destination_fingerprint(requested_url),
    )


def access_error_result(
    evidence: WebAccessEvidence,
    *,
    error: str = "access_blocked",
    message: str | None = None,
    effective_source_url: str | None = None,
) -> ToolResult:
    structured: dict[str, Any] = {
        "error": error,
        "access": evidence.model_dump(mode="json"),
    }
    if evidence.status_code is not None:
        structured["status_code"] = evidence.status_code
    if effective_source_url is not None:
        effective_origin = _safe_effective_origin(effective_source_url)
        if effective_origin is None or (
            web_destination_fingerprint(effective_origin) != evidence.destination_fingerprint
        ):
            raise ValueError("Access evidence conflicts with its effective origin.")
        structured["effective_source_url"] = effective_origin
    return ToolResult(
        content=_ACCESS_MESSAGES[evidence.outcome] if message is None else message,
        structured=structured,
        is_error=True,
    )


def attach_access_evidence(
    result: ToolResult,
    evidence: WebAccessEvidence,
) -> ToolResult:
    """Attach validated access evidence without retaining denial-page content."""

    source = result.structured if isinstance(result.structured, Mapping) else {}
    error = source.get("error")
    structured: dict[str, Any] = {
        "error": (
            error if type(error) is str and error in _SAFE_ACCESS_ERROR_CODES else "access_blocked"
        ),
        "access": evidence.model_dump(mode="json"),
    }
    if evidence.status_code is not None:
        structured["status_code"] = evidence.status_code
    effective_origin = _safe_effective_origin(source.get("effective_source_url"))
    if effective_origin is None:
        effective_origin = _safe_effective_origin(source.get("final_url"))
    if effective_origin is not None:
        if web_destination_fingerprint(effective_origin) != evidence.destination_fingerprint:
            raise ValueError("Access evidence conflicts with its effective origin.")
        structured["effective_source_url"] = effective_origin
    if result.is_error:
        return ToolResult(
            content=_ACCESS_MESSAGES[evidence.outcome],
            structured=structured,
            is_error=True,
        )
    raise ValueError("Access evidence may be attached only to an error result.")


def access_evidence_from_result(
    result: ToolResult,
    *,
    requested_url: str,
    source: WebAccessEvidenceSource = WebAccessEvidenceSource.HOSTED_PROVIDER,
    allowed_sources: frozenset[WebAccessEvidenceSource] | None = None,
) -> WebAccessEvidence | None:
    if not result.is_error:
        return None
    structured = result.structured
    if not isinstance(structured, Mapping):
        return None
    raw_access = structured.get("access")
    if raw_access is not None:
        try:
            evidence = WebAccessEvidence.model_validate(raw_access)
        except (TypeError, ValueError):
            return None
        expected_sources = frozenset({source}) if allowed_sources is None else allowed_sources
        effective_origins: set[str] = set()
        for field in ("effective_source_url", "final_url"):
            candidate = structured.get(field)
            if candidate is None:
                continue
            effective_origin = _safe_effective_origin(candidate)
            if effective_origin is None:
                return None
            effective_origins.add(effective_origin)
        if len(effective_origins) > 1:
            return None
        expected_destination = web_destination_fingerprint(
            next(iter(effective_origins)) if effective_origins else requested_url
        )
        if (
            evidence.source not in expected_sources
            or evidence.destination_fingerprint != expected_destination
        ):
            return None
        return evidence
    evidence_url = _result_effective_source_origin(result)
    if evidence_url is None:
        evidence_url = requested_url
    error = structured.get("error")
    error_key = error if type(error) is str else ""
    status_code = structured.get("status_code")
    priority_mapping = {
        "provider_authentication_failed": WebAccessOutcome.AUTHENTICATION_REQUIRED,
        "rate_limited": WebAccessOutcome.RATE_LIMITED,
    }
    outcome = priority_mapping.get(error_key)
    if outcome is None and type(status_code) is int:
        classified = classify_http_access(
            evidence_url,
            status_code=status_code,
            headers={},
            source=source,
        )
        if classified is not None:
            return classified
    mapping = {
        "provider_unavailable": WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE,
        "timeout": WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE,
    }
    if outcome is None:
        outcome = mapping.get(error_key)
    if outcome is not None:
        retry_after, retry_after_unrepresentable = (
            _provider_retry_after(structured)
            if outcome is WebAccessOutcome.RATE_LIMITED
            else (None, False)
        )
        try:
            return WebAccessEvidence(
                outcome=outcome,
                source=source,
                signal=(
                    WebAccessSignal.RETRY_AFTER
                    if retry_after is not None or retry_after_unrepresentable
                    else WebAccessSignal.PROVIDER_STATUS
                ),
                destination_fingerprint=web_destination_fingerprint(evidence_url),
                status_code=status_code if type(status_code) is int else None,
                retry_after_seconds=retry_after,
                retry_after_unrepresentable=retry_after_unrepresentable,
            )
        except (TypeError, ValueError):
            return None
    if error_key in {"destination_denied", "redirect_denied", "policy_denied"}:
        expected_sources = frozenset({source}) if allowed_sources is None else allowed_sources
        if (
            source is not WebAccessEvidenceSource.BROWSER_RESPONSE
            or WebAccessEvidenceSource.EGRESS_POLICY not in expected_sources
        ):
            return None
        return transport_access_evidence(
            evidence_url,
            outcome=WebAccessOutcome.DESTINATION_DENIED,
            source=WebAccessEvidenceSource.EGRESS_POLICY,
            signal=WebAccessSignal.EGRESS_DENIAL,
        )
    return None


def web_destination_fingerprint(url: str) -> str:
    canonical = _requested_url({"url": url})
    if canonical is None:
        raise ValueError("Web destination must be a canonical HTTPS URL.")
    split = urlsplit(canonical)
    if split.hostname is None:  # pragma: no cover - canonicalization requires a host
        raise ValueError("Web destination must be a canonical HTTPS URL.")
    origin = f"https://{split.hostname}/"
    return hashlib.sha256(b"cayu.web-access-destination.v1\0" + origin.encode("utf-8")).hexdigest()


_SAFE_ACCESS_ERROR_CODES = frozenset(
    {
        "access_blocked",
        "browser_crash",
        "destination_denied",
        "dns_failure",
        "fetch_failed",
        "http_status",
        "policy_denied",
        "provider_authentication_failed",
        "provider_unavailable",
        "rate_limited",
        "redirect_denied",
        "timeout",
    }
)

_ACCESS_MESSAGES = {
    WebAccessOutcome.AUTHENTICATION_REQUIRED: (
        "The destination requires application-approved authentication."
    ),
    WebAccessOutcome.CONSENT_REQUIRED: "The destination requires operator consent.",
    WebAccessOutcome.RATE_LIMITED: "The destination rate-limited this access route.",
    WebAccessOutcome.BOT_CHALLENGE: (
        "The destination requires an interactive anti-bot challenge; Cayu did not solve it."
    ),
    WebAccessOutcome.DESTINATION_DENIED: "The destination was denied by access policy.",
    WebAccessOutcome.CONTENT_UNAVAILABLE: "The requested content is unavailable.",
    WebAccessOutcome.TRANSIENT_TRANSPORT_FAILURE: (
        "The selected route encountered a transient transport failure."
    ),
}


def _route_id(value: str) -> str:
    value = require_durable_clean_nonblank(value, "route_id")
    if (
        len(value.encode("utf-8")) > MAX_WEB_ACCESS_ROUTE_ID_BYTES
        or _ROUTE_ID.fullmatch(value) is None
    ):
        raise ValueError("route_id must be a lowercase bounded identifier.")
    return value


def _route_fetch_tool(route: WebBridgeRoute) -> Tool:
    return next(tool for tool in route.bridge.tools if getattr(tool, "name", None) == "web_fetch")


def _route_fetch_material(route: WebBridgeRoute) -> dict[str, Any] | None:
    fetch_tool = _route_fetch_tool(route)
    material_method = getattr(fetch_tool, "_execution_profile_material", None)
    if callable(material_method):
        material = material_method()
        if material is not None:
            return {"authority": "cayu_owned", "material": material}
    identity = fetch_tool.execution_profile_identity
    if identity is None:
        return None
    wrapper_method = getattr(fetch_tool, "_execution_profile_wrapper_material", None)
    if not callable(wrapper_method):
        return None
    wrapper_material = wrapper_method()
    if type(wrapper_material) is not dict:
        return None
    return {
        "authority": "application_versioned",
        "wrapper": wrapper_material,
        "identity": identity.model_dump(mode="json"),
    }


def _route_invocation_context(
    ctx: ToolContext,
    *,
    policy_fingerprint: str,
    route_id: str,
) -> ToolContext:
    outer_key = ctx.idempotency_key
    if type(outer_key) is not str or not outer_key:
        raise RuntimeError("Web access routing requires an idempotency identity.")
    digest = hashlib.sha256(
        canonical_durable_json_bytes(
            [outer_key, policy_fingerprint, route_id],
            "web route idempotency identity",
        )
    ).hexdigest()
    downstream_key = f"cayu-web-route:v1:{digest}"
    metadata = dict(ctx.metadata)
    metadata["idempotency_key"] = downstream_key
    return ctx.model_copy(
        update={
            "idempotency_key": downstream_key,
            "metadata": metadata,
        }
    )


def _route_access_sources(
    route: WebBridgeRoute,
) -> tuple[WebAccessEvidenceSource, frozenset[WebAccessEvidenceSource]]:
    from cayu.tools.webbridge import WebBridgeProfileKind

    primary = {
        WebBridgeProfileKind.TRUSTED_LOCAL: WebAccessEvidenceSource.HTTP_RESPONSE,
        WebBridgeProfileKind.HOSTED_PROVIDER: WebAccessEvidenceSource.HOSTED_PROVIDER,
        WebBridgeProfileKind.SANDBOXED_BROWSER: WebAccessEvidenceSource.BROWSER_RESPONSE,
    }.get(route.bridge.kind)
    if primary is None:
        raise ValueError("A routed WebBridge contains an unsupported profile.")
    return (
        primary,
        frozenset(
            {
                primary,
                WebAccessEvidenceSource.EGRESS_POLICY,
                WebAccessEvidenceSource.TRANSPORT,
            }
        ),
    )


def _bridge_route_identity(route: WebBridgeRoute) -> dict[str, str]:
    bridge = route.bridge
    authority = bridge.credential_authority
    fetch_material = _route_fetch_material(route)
    if fetch_material is None:
        raise ValueError("Routed hosted WebBridges require an explicit execution_profile_identity.")
    material = {
        "route_id": route.route_id,
        "kind": bridge.kind.value,
        "execution_location": bridge.execution_location,
        "credential_path": bridge.credential_path,
        "workspace_requirement": bridge.workspace_requirement,
        "artifact_store_id": bridge.artifact_store_id,
        "browser_protocol": bridge.browser_protocol,
        "browser_worker_version": bridge.browser_worker_version,
        "playwright_version": bridge.playwright_version,
        "fetch_authority": fetch_material,
        "credential_authority": (
            None
            if authority is None
            else {
                "provider": authority.provider,
                "origin": authority.origin,
                "secret_refs_sha256": hashlib.sha256(
                    b"cayu.webbridge-credential-authority.v1\0"
                    + canonical_durable_json_bytes(
                        [ref.model_dump(mode="json") for ref in authority.secret_refs],
                        "credential_authority",
                    )
                ).hexdigest(),
            }
        ),
    }
    profile_fingerprint = hashlib.sha256(
        b"cayu.webbridge-route.v1\0" + canonical_durable_json_bytes(material, "route_profile")
    ).hexdigest()
    return {
        "route_id": route.route_id,
        "kind": bridge.kind.value,
        "profile_fingerprint": profile_fingerprint,
    }


def _policy_fingerprint(
    policy: WebAccessRoutePolicy,
    identities: Mapping[str, Mapping[str, str]],
) -> str:
    material = {
        "entry_route_id": policy.entry_route_id,
        "routes": [dict(identities[key]) for key in sorted(identities)],
        "rules": [
            {
                "route_id": rule.route_id,
                "outcome": rule.outcome.value,
                "action": rule.action.kind.value,
                "target_route_id": rule.action.target_route_id,
                "wait_seconds": rule.action.wait_seconds,
                "guidance": rule.action.guidance,
            }
            for rule in policy.rules
        ],
        "circuit": {
            "threshold": policy.circuit.threshold,
            "open_seconds": policy.circuit.open_seconds,
            "max_entries": policy.circuit.max_entries,
        },
    }
    return hashlib.sha256(
        b"cayu.web-access-policy.v1\0" + canonical_durable_json_bytes(material, "route_policy")
    ).hexdigest()


def _require_acyclic_fallbacks(rules: Sequence[WebAccessRouteRule]) -> None:
    graph: dict[str, set[str]] = {}
    for rule in rules:
        target = rule.action.target_route_id
        if target is not None:
            graph.setdefault(rule.route_id, set()).add(target)

    def visit(node: str, active: set[str], complete: set[str]) -> None:
        if node in active:
            raise ValueError("Fallback routes must form an acyclic graph.")
        if node in complete:
            return
        active.add(node)
        for target in graph.get(node, ()):
            visit(target, active, complete)
        active.remove(node)
        complete.add(node)

    complete: set[str] = set()
    for node in graph:
        visit(node, set(), complete)


def _requested_url(args: Any) -> str | None:
    if type(args) is not dict or set(args) != {"url"} or type(args.get("url")) is not str:
        return None
    value = args["url"]
    if len(value) > 8192:
        return None
    try:
        from cayu.tools.web import _canonicalize_url

        return _canonicalize_url(value)
    except (TypeError, ValueError):
        return None


def _safe_effective_origin(value: Any) -> str | None:
    if type(value) is not str:
        return None
    canonical = _requested_url({"url": value})
    if canonical is None:
        return None
    split = urlsplit(canonical)
    if split.hostname is None:
        return None
    return f"https://{split.hostname.lower()}/"


def _result_effective_source_origin(result: ToolResult) -> str | None:
    structured = result.structured
    if not isinstance(structured, Mapping):
        return None
    effective_origin = _safe_effective_origin(structured.get("effective_source_url"))
    if effective_origin is None:
        effective_origin = _safe_effective_origin(structured.get("final_url"))
    return effective_origin


def _result_effective_source_url(
    result: ToolResult,
    *,
    default: str,
) -> str:
    structured = result.structured
    if isinstance(structured, Mapping):
        value = structured.get("final_url")
        if type(value) is str:
            canonical = _requested_url({"url": value})
            if canonical is not None:
                return canonical
    canonical_default = _requested_url({"url": default})
    if canonical_default is None:
        raise ValueError("The routed request lost its effective source URL.")
    return canonical_default


def _routing_authority(
    ctx: ToolContext,
    args: dict[str, Any],
) -> _RuntimeToolInvocationAuthority | None:
    authority = _runtime_tool_invocation_authority(ctx)
    if authority is None:
        return None
    digest = hashlib.sha256(canonical_durable_json_bytes(args, "web_access_arguments")).hexdigest()
    if (
        authority.tool_name != "web_fetch"
        or authority.idempotency_key != ctx.idempotency_key
        or authority.effective_arguments_sha256 != digest
    ):
        raise RuntimeError("Web access routing authority conflicts with its invocation.")
    return authority


def _denial_fingerprint(
    route_id: str,
    request_destination_fingerprint: str,
    evidence: WebAccessEvidence,
) -> str:
    return hashlib.sha256(
        b"cayu.web-access-denial.v1\0"
        + canonical_durable_json_bytes(
            [
                route_id,
                request_destination_fingerprint,
                evidence.destination_fingerprint,
                evidence.outcome.value,
            ],
            "denial_identity",
        )
    ).hexdigest()


def _empty_circuit_record(policy_fingerprint: str) -> dict[str, Any]:
    return {
        "record_type": _CIRCUIT_RECORD_TYPE,
        "schema_version": _CIRCUIT_SCHEMA_VERSION,
        "policy_fingerprint": policy_fingerprint,
        "entries": [],
    }


def _validated_circuit_record(
    raw: Mapping[str, Any] | None,
    *,
    policy_fingerprint: str,
    route_ids: frozenset[str],
    circuit: WebAccessCircuitPolicy,
) -> dict[str, Any]:
    if raw is None:
        return _empty_circuit_record(policy_fingerprint)
    record = copy_durable_json_object(raw, "web_access_circuit")
    if (
        record.get("record_type") != _CIRCUIT_RECORD_TYPE
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != _CIRCUIT_SCHEMA_VERSION
        or record.get("policy_fingerprint") != policy_fingerprint
        or type(record.get("entries")) is not list
        or len(record["entries"]) > circuit.max_entries
    ):
        raise RuntimeError("Durable web-access circuit authority is incompatible.")
    seen: set[str] = set()
    for entry in record["entries"]:
        if type(entry) is not dict or set(entry) != _CIRCUIT_ENTRY_FIELDS:
            raise RuntimeError("Durable web-access circuit evidence is malformed.")
        if (
            type(entry["authority_sha256"]) is not str
            or _SHA256.fullmatch(entry["authority_sha256"]) is None
            or type(entry["fingerprint"]) is not str
            or _SHA256.fullmatch(entry["fingerprint"]) is None
            or entry["fingerprint"] in seen
            or type(entry["route_id"]) is not str
            or _ROUTE_ID.fullmatch(entry["route_id"]) is None
            or entry["route_id"] not in route_ids
            or type(entry["request_destination_fingerprint"]) is not str
            or _SHA256.fullmatch(entry["request_destination_fingerprint"]) is None
            or type(entry["destination_fingerprint"]) is not str
            or _SHA256.fullmatch(entry["destination_fingerprint"]) is None
            or (
                entry["effective_source_url"] is not None
                and (
                    _safe_effective_origin(entry["effective_source_url"])
                    != entry["effective_source_url"]
                    or web_destination_fingerprint(entry["effective_source_url"])
                    != entry["destination_fingerprint"]
                )
            )
            or entry["outcome"] not in {outcome.value for outcome in WebAccessOutcome}
            or entry["source"] not in {source.value for source in WebAccessEvidenceSource}
            or entry["signal"] not in {signal.value for signal in WebAccessSignal}
            or (
                entry["status_code"] is not None
                and (
                    type(entry["status_code"]) is not int or not 100 <= entry["status_code"] <= 599
                )
            )
            or (
                entry["retry_after_seconds"] is not None
                and (
                    type(entry["retry_after_seconds"]) is not int
                    or not 0 <= entry["retry_after_seconds"] <= MAX_WEB_ACCESS_RETRY_AFTER_SECONDS
                )
            )
            or (
                entry["outcome"] != WebAccessOutcome.RATE_LIMITED
                and entry["retry_after_seconds"] is not None
            )
            or type(entry["retry_after_unrepresentable"]) is not bool
            or (
                entry["retry_after_unrepresentable"]
                and (
                    entry["outcome"] != WebAccessOutcome.RATE_LIMITED
                    or entry["signal"] != WebAccessSignal.RETRY_AFTER
                    or entry["retry_after_seconds"] is not None
                    or entry["next_eligible_at"] is not None
                )
            )
            or type(entry["denial_count"]) is not int
            or not 1 <= entry["denial_count"] <= circuit.threshold
            or (
                entry["next_eligible_at"] is not None
                and (type(entry["next_eligible_at"]) is not int or entry["next_eligible_at"] < 0)
            )
            or type(entry["updated_at"]) is not int
            or entry["updated_at"] < 0
        ):
            raise RuntimeError("Durable web-access circuit evidence is malformed.")
        try:
            evidence = WebAccessEvidence(
                outcome=WebAccessOutcome(entry["outcome"]),
                source=WebAccessEvidenceSource(entry["source"]),
                signal=WebAccessSignal(entry["signal"]),
                destination_fingerprint=entry["destination_fingerprint"],
                status_code=entry["status_code"],
                retry_after_seconds=entry["retry_after_seconds"],
                retry_after_unrepresentable=entry["retry_after_unrepresentable"],
            )
            expected_fingerprint = _denial_fingerprint(
                entry["route_id"],
                entry["request_destination_fingerprint"],
                evidence,
            )
            _timestamp(entry["updated_at"])
            if entry["next_eligible_at"] is not None:
                _timestamp(entry["next_eligible_at"])
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise RuntimeError("Durable web-access circuit evidence is malformed.") from exc
        if (
            entry["fingerprint"] != expected_fingerprint
            or entry["authority_sha256"] != _circuit_entry_authority_sha256(entry)
            or (
                entry["next_eligible_at"] is not None
                and entry["next_eligible_at"] < entry["updated_at"]
            )
        ):
            raise RuntimeError("Durable web-access circuit evidence is malformed.")
        seen.add(entry["fingerprint"])
    if [entry["fingerprint"] for entry in record["entries"]] != sorted(seen):
        raise RuntimeError("Durable web-access circuit evidence is malformed.")
    return record


def _seal_circuit_record(
    authority: _RuntimeToolInvocationAuthority,
    desired: dict[str, Any],
    *,
    policy_fingerprint: str,
    route_ids: frozenset[str],
    circuit: WebAccessCircuitPolicy,
) -> dict[str, Any]:
    """Seal untrusted values without redacting runtime-owned circuit controls."""

    sealed = authority.seal_durable_output(desired)
    if type(sealed) is not dict:
        raise RuntimeError("Durable web-access circuit sealing returned invalid evidence.")
    original_entries = desired.get("entries")
    sealed_entries = sealed.get("entries")
    if (
        type(original_entries) is not list
        or type(sealed_entries) is not list
        or len(original_entries) != len(sealed_entries)
    ):
        raise RuntimeError("Durable web-access circuit sealing returned invalid evidence.")
    sealed["record_type"] = desired["record_type"]
    sealed["schema_version"] = desired["schema_version"]
    sealed["policy_fingerprint"] = desired["policy_fingerprint"]
    for original_entry, sealed_entry in zip(original_entries, sealed_entries, strict=True):
        if type(original_entry) is not dict or type(sealed_entry) is not dict:
            raise RuntimeError("Durable web-access circuit sealing returned invalid evidence.")
        owned_original_entry = cast("dict[str, Any]", original_entry)
        owned_sealed_entry = cast("dict[str, Any]", sealed_entry)
        for field, value in owned_original_entry.items():
            if field not in {"authority_sha256", "effective_source_url"}:
                owned_sealed_entry[field] = value
        if owned_sealed_entry.get("effective_source_url") != owned_original_entry.get(
            "effective_source_url"
        ):
            owned_sealed_entry["effective_source_url"] = None
        owned_sealed_entry["authority_sha256"] = _circuit_entry_authority_sha256(owned_sealed_entry)
    return _validated_circuit_record(
        sealed,
        policy_fingerprint=policy_fingerprint,
        route_ids=route_ids,
        circuit=circuit,
    )


def _prepare_intermediate_circuit_record(
    desired: dict[str, Any],
    *,
    pending_effective_origins: Mapping[str, str],
    policy_fingerprint: str,
    route_ids: frozenset[str],
    circuit: WebAccessCircuitPolicy,
) -> dict[str, Any]:
    """Prepare content-free circuit authority without sealing later route secrets."""

    prepared = copy_durable_json_object(desired, "web_access_intermediate_circuit")
    entries = prepared.get("entries")
    if type(entries) is not list:
        raise RuntimeError("Durable web-access circuit evidence is malformed.")
    for entry in entries:
        if type(entry) is not dict:
            raise RuntimeError("Durable web-access circuit evidence is malformed.")
        if entry.get("fingerprint") in pending_effective_origins:
            entry["effective_source_url"] = None
        entry["authority_sha256"] = _circuit_entry_authority_sha256(entry)
    return _validated_circuit_record(
        prepared,
        policy_fingerprint=policy_fingerprint,
        route_ids=route_ids,
        circuit=circuit,
    )


def _apply_pending_effective_origins(
    entries: list[_CircuitEntry],
    pending_effective_origins: Mapping[str, str],
) -> None:
    for entry in entries:
        effective_origin = pending_effective_origins.get(entry["fingerprint"])
        if effective_origin is not None:
            entry["effective_source_url"] = effective_origin


def _circuit_entry_authority_sha256(entry: Mapping[str, Any]) -> str:
    if not isinstance(entry, Mapping):
        raise RuntimeError("Durable web-access circuit evidence is malformed.")
    try:
        material = {
            field: entry[field] for field in _CIRCUIT_ENTRY_FIELDS if field != "authority_sha256"
        }
    except KeyError as exc:
        raise RuntimeError("Durable web-access circuit evidence is malformed.") from exc
    return hashlib.sha256(
        b"cayu.web-access-circuit-entry.v1\0"
        + canonical_durable_json_bytes(material, "web_access_circuit_entry")
    ).hexdigest()


def _history_entry(
    *,
    route_identity: Mapping[str, str],
    evidence: WebAccessEvidence,
    action: WebAccessRouteAction | None,
    invoked: bool,
    next_eligible_at: int | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "route": dict(route_identity),
        "invoked": invoked,
        "access": evidence.model_dump(mode="json"),
        "action": "stop" if action is None else action.kind.value,
    }
    if next_eligible_at is not None:
        entry["next_eligible_at"] = _timestamp(next_eligible_at)
    return entry


def _route_decision(
    action: WebAccessRouteAction | None,
    *,
    evidence: WebAccessEvidence,
    circuit_next_eligible_at: int | None,
    now: int,
) -> tuple[str, str, int | None]:
    if action is None or action.kind is WebAccessRouteActionKind.STOP:
        return "stopped", "", circuit_next_eligible_at
    if action.kind is WebAccessRouteActionKind.FALLBACK:
        return "fallback", action.target_route_id or "", circuit_next_eligible_at
    if action.kind is WebAccessRouteActionKind.OPERATOR_ACTION:
        return "operator_action", "", circuit_next_eligible_at
    if evidence.retry_after_unrepresentable:
        return "operator_action", "", None
    wait_until = now + (action.wait_seconds or 0)
    if circuit_next_eligible_at is not None:
        wait_until = max(wait_until, circuit_next_eligible_at)
    return "wait", "", wait_until


def _access_terminal_result(
    *,
    evidence: WebAccessEvidence,
    action: WebAccessRouteAction | None,
    history: Sequence[Mapping[str, Any]],
    original: WebAccessEvidence,
    selected: Mapping[str, str],
    execution_profile_fingerprint: str,
    disposition: str,
    next_eligible_at: int | None,
    effective_source_url: str | None,
) -> ToolResult:
    guidance = (
        "Authoritative retry timing exceeds Cayu's bounded horizon; use another explicitly "
        "permitted route or ask an operator to stop."
        if evidence.retry_after_unrepresentable
        else (None if action is None else action.guidance)
    )
    if guidance is None:
        guidance = _default_guidance(evidence.outcome, disposition)
    structured: dict[str, Any] = {
        "error": "access_blocked",
        "access": evidence.model_dump(mode="json"),
    }
    return _routing_terminal(
        result=ToolResult(content=guidance, structured=structured, is_error=True),
        history=history,
        original=original,
        selected=selected,
        execution_profile_fingerprint=execution_profile_fingerprint,
        disposition=disposition,
        next_eligible_at=next_eligible_at,
        guidance=guidance,
        effective_source_url=effective_source_url,
    )


def _routing_terminal(
    *,
    result: ToolResult,
    history: Sequence[Mapping[str, Any]],
    original: WebAccessEvidence | None,
    selected: Mapping[str, str],
    execution_profile_fingerprint: str,
    disposition: str,
    next_eligible_at: int | None = None,
    guidance: str | None = None,
    default_effective_source_url: str | None = None,
    effective_source_url: str | None = None,
) -> ToolResult:
    structured = {} if result.structured is None else dict(result.structured)
    route: dict[str, Any] = {
        "schema_version": 1,
        "policy": "explicit",
        "selected_route": dict(selected),
        "execution_profile_fingerprint": execution_profile_fingerprint,
        "terminal_disposition": disposition,
        "history": [dict(item) for item in history],
    }
    if original is not None:
        route["original_access"] = original.model_dump(mode="json")
    if next_eligible_at is not None:
        route["next_eligible_at"] = _timestamp(next_eligible_at)
    if guidance is not None:
        route["guidance"] = guidance
    if effective_source_url is not None:
        route["effective_source_url"] = effective_source_url
    elif not result.is_error:
        effective_url = structured.get("final_url")
        if type(effective_url) is not str or _requested_url({"url": effective_url}) is None:
            effective_url = default_effective_source_url
        if effective_url is not None:
            route["effective_source_url"] = effective_url
    elif original is not None and default_effective_source_url is not None:
        effective_origin = _safe_effective_origin(default_effective_source_url)
        if effective_origin is not None:
            route["effective_source_url"] = effective_origin
    structured["webbridge_route"] = route
    prefix = ""
    if disposition == "fallback_succeeded":
        prefix = "An explicit WebBridge fallback produced replacement evidence.\n\n"
    return ToolResult(
        content=prefix + result.content,
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.is_error,
    )


def _default_guidance(outcome: WebAccessOutcome, disposition: str) -> str:
    if outcome is WebAccessOutcome.AUTHENTICATION_REQUIRED:
        return "Configure approved credentials for a permitted route or ask an operator to stop."
    if outcome in {WebAccessOutcome.CONSENT_REQUIRED, WebAccessOutcome.BOT_CHALLENGE}:
        return "Operator action is required; Cayu did not bypass or solve the site control."
    if disposition == "wait":
        return "Retry only after next_eligible_at and within the caller's cumulative budget."
    return "No explicitly permitted replacement route completed this request."


def _plain_error(code: str, message: str) -> ToolResult:
    return ToolResult(content=message, structured={"error": code}, is_error=True)


def _bounded_access_headers(headers: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        raise TypeError("headers must be a mapping.")
    allowed = {
        "cf-mitigated",
        "retry-after",
        "www-authenticate",
        "x-cayu-access-block",
        "x-cayu-access-requirement",
    }
    output: dict[str, str] = {}
    for key, value in headers.items():
        if type(key) is not str or type(value) is not str:
            continue
        lowered = key.lower()
        if lowered in allowed:
            encoded = value.encode("utf-8", errors="replace")
            output[lowered] = encoded[:256].decode("utf-8", errors="ignore")
    return output


def _retry_after_seconds(
    value: str | None,
    *,
    now: datetime | None = None,
) -> tuple[int | None, bool]:
    if value is None:
        return None, False
    stripped = value.strip()
    if not stripped:
        return None, False
    if stripped.isascii() and stripped.isdigit():
        if len(stripped) > 128:
            return None, True
        canonical_digits = stripped.lstrip("0") or "0"
        if len(canonical_digits) > 5:
            return None, True
        seconds = int(canonical_digits)
        if seconds > MAX_WEB_ACCESS_RETRY_AFTER_SECONDS:
            return None, True
        return seconds, False
    if len(stripped) > 128:
        return None, False
    try:
        target = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None, False
    if target.tzinfo is None:
        return None, False
    relative = datetime.now(UTC) if now is None else now.astimezone(UTC)
    delta = math.ceil((target.astimezone(UTC) - relative).total_seconds())
    if delta > MAX_WEB_ACCESS_RETRY_AFTER_SECONDS:
        return None, True
    if delta >= 0:
        return delta, False
    return None, False


def _provider_retry_after(structured: Mapping[str, Any]) -> tuple[int | None, bool]:
    provider_metadata = structured.get("provider_metadata")
    if not isinstance(provider_metadata, Mapping):
        return None, False
    bounded: list[int] = []
    unrepresentable_observed = False
    for value in provider_metadata.values():
        if not isinstance(value, Mapping):
            continue
        unrepresentable = value.get("retry_after_unrepresentable")
        if type(unrepresentable) is bool and unrepresentable:
            unrepresentable_observed = True
        retry = value.get("retry_after_seconds")
        if type(retry) is int:
            if retry > MAX_WEB_ACCESS_RETRY_AFTER_SECONDS:
                unrepresentable_observed = True
            elif retry >= 0:
                bounded.append(retry)
        if type(retry) is float:
            if not math.isfinite(retry) or retry > MAX_WEB_ACCESS_RETRY_AFTER_SECONDS:
                unrepresentable_observed = True
            elif retry >= 0:
                bounded.append(math.ceil(retry))
    if unrepresentable_observed:
        return None, True
    return (max(bounded), False) if bounded else (None, False)


def _utc_now_seconds() -> int:
    return math.floor(datetime.now(UTC).timestamp())


def _timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "MAX_WEB_ACCESS_CIRCUIT_ENTRIES",
    "MAX_WEB_ACCESS_RETRY_AFTER_SECONDS",
    "MAX_WEB_ACCESS_ROUTES",
    "WebAccessCircuitPolicy",
    "WebAccessEvidence",
    "WebAccessEvidenceSource",
    "WebAccessOutcome",
    "WebAccessRouteAction",
    "WebAccessRouteActionKind",
    "WebAccessRoutePolicy",
    "WebAccessRouteRule",
    "WebAccessRoutingTool",
    "WebAccessSignal",
    "WebBridgeRoute",
    "access_error_result",
    "access_evidence_from_result",
    "attach_access_evidence",
    "classify_http_access",
    "transport_access_evidence",
    "web_destination_fingerprint",
]
