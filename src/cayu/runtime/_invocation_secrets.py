"""Invocation-scoped secret tracking and authenticated cancellation handoff."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from cayu._exception_state import exception_state, pop_exception_state, set_exception_state
from cayu._validation import (
    copy_durable_json_object,
    copy_json_object,
    copy_json_value,
    require_clean_nonblank,
    require_durable_text,
)
from cayu.proxies import (
    CredentialProxy,
    ProxyAuthorizationResult,
    copy_proxy_authorization_result,
)
from cayu.runners import RunnerCancelledError
from cayu.runtime import _runtime_records as runtime_records
from cayu.vaults import (
    ResolvedSecret,
    SecretRedactor,
    SecretRef,
    Vault,
    copy_resolved_secret,
    copy_secret_ref,
)

_CANCELLATION_EVIDENCE_ATTRIBUTE = "_cayu_tool_cancellation_evidence"
_CANCELLATION_EVIDENCE_TOKEN = object()
_LEGACY_CANCELLATION_TOOL_CALL_ID_ATTRIBUTE = "_cayu_cancellation_tool_call_id"
_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_BASE_EXCEPTION_DICT_DESCRIPTOR = BaseException.__dict__["__dict__"]


@dataclass(frozen=True, slots=True)
class ProxyAuthorizationRecord:
    destination: str
    credential: SecretRef | None
    action: str | None
    metadata: dict[str, Any]
    result: ProxyAuthorizationResult


class InvocationSecretTracker:
    """Own the secret-redaction scope for one tool invocation."""

    def __init__(self, redactor: SecretRedactor) -> None:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        self._redactor = redactor

    @property
    def redactor(self) -> SecretRedactor:
        return self._redactor

    def record(self, secret: ResolvedSecret) -> None:
        if type(secret) is not ResolvedSecret:
            raise TypeError("Resolved secrets must be ResolvedSecret instances.")
        self._redactor = self._redactor.with_secret(secret)


class _TrackingVault(Vault):
    """Track secrets resolved directly through a tool's vault boundary."""

    def __init__(
        self,
        vault: Vault,
        on_resolve: Callable[[ResolvedSecret], None],
    ) -> None:
        if not isinstance(vault, Vault):
            raise TypeError("vault must be a Vault.")
        if not callable(on_resolve):
            raise TypeError("on_resolve must be callable.")
        self._vault = vault
        self._on_resolve = on_resolve

    async def get(
        self,
        name: str,
        *,
        scope: dict[str, Any] | None = None,
    ) -> SecretRef:
        copied_scope = None if scope is None else copy_json_object(scope, "scope")
        ref = await self._vault.get(
            require_clean_nonblank(name, "name"),
            scope=None if copied_scope is None else copy_json_object(copied_scope, "scope"),
        )
        if type(ref) is not SecretRef:
            raise TypeError("Vault lookup must return SecretRef.")
        return copy_secret_ref(ref)

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        copied_ref = copy_secret_ref(ref)
        copied_scope = None if scope is None else copy_json_object(scope, "scope")
        secret = await self._vault.resolve(
            copied_ref,
            scope=None if copied_scope is None else copy_json_object(copied_scope, "scope"),
        )
        if type(secret) is not ResolvedSecret:
            raise TypeError("Vault secret resolution must return ResolvedSecret.")
        self._on_resolve(copy_resolved_secret(secret))
        return copy_resolved_secret(secret)


class _TrackingCredentialProxy(CredentialProxy):
    def __init__(
        self,
        proxy: CredentialProxy,
        on_resolve: Callable[[ResolvedSecret], None],
        on_authorize: Callable[[ProxyAuthorizationRecord], None],
    ) -> None:
        if not isinstance(proxy, CredentialProxy):
            raise TypeError("proxy must be a CredentialProxy.")
        if not callable(on_resolve):
            raise TypeError("on_resolve must be callable.")
        if not callable(on_authorize):
            raise TypeError("on_authorize must be callable.")
        self._proxy = proxy
        self._on_resolve = on_resolve
        self._on_authorize = on_authorize

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        copied_ref = copy_secret_ref(ref)
        copied_scope = None if scope is None else copy_json_object(scope, "scope")
        secret = await self._proxy.resolve(
            copied_ref,
            scope=None if copied_scope is None else copy_json_object(copied_scope, "scope"),
        )
        if type(secret) is not ResolvedSecret:
            raise TypeError("Proxy secret resolution must return ResolvedSecret.")
        self._on_resolve(copy_resolved_secret(secret))
        return copy_resolved_secret(secret)

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        copied_destination = require_clean_nonblank(
            require_durable_text(destination, "destination"),
            "destination",
        )
        copied_credential = (
            None if credential is None else _copy_durable_proxy_secret_ref(credential)
        )
        copied_action = (
            None
            if action is None
            else require_clean_nonblank(
                require_durable_text(action, "action"),
                "action",
            )
        )
        copied_metadata = {} if metadata is None else copy_durable_json_object(metadata, "metadata")
        result = await self._proxy.authorize_request(
            destination=copied_destination,
            credential=(
                None
                if copied_credential is None
                else _copy_durable_proxy_secret_ref(copied_credential)
            ),
            action=copied_action,
            metadata=copy_durable_json_object(copied_metadata, "metadata"),
        )
        if type(result) is not ProxyAuthorizationResult:
            raise TypeError("Proxy authorization must return ProxyAuthorizationResult.")
        copied_result = _copy_durable_proxy_authorization_result(result)
        self._on_authorize(
            ProxyAuthorizationRecord(
                destination=copied_destination,
                credential=copied_credential,
                action=copied_action,
                metadata=copied_metadata,
                result=copied_result,
            )
        )
        return _copy_durable_proxy_authorization_result(copied_result)


def vault_for_environment(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    on_resolve: Callable[[ResolvedSecret], None],
) -> Vault | None:
    if registered_environment is None:
        return None
    vault = registered_environment.environment.vault
    if vault is None:
        return None
    return _TrackingVault(vault, on_resolve)


def proxy_for_environment(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    on_resolve: Callable[[ResolvedSecret], None],
    on_authorize: Callable[[ProxyAuthorizationRecord], None],
) -> CredentialProxy | None:
    if registered_environment is None:
        return None
    proxy = registered_environment.environment.proxy
    if proxy is None:
        return None
    return _TrackingCredentialProxy(proxy, on_resolve, on_authorize)


@dataclass(frozen=True, slots=True)
class _ToolCancellationEvidence:
    token: object
    tool_call_id: str | None = None
    redactor: SecretRedactor | None = None
    artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None
    redactors_by_id: dict[str, SecretRedactor] | None = None


def initialize_cancellation_evidence(cancellation: asyncio.CancelledError) -> None:
    """Replace caller-supplied lookalikes with one authenticated runtime handoff."""

    _clear_legacy_cancellation_tool_call_id(cancellation)
    if not set_exception_state(
        cancellation,
        _CANCELLATION_EVIDENCE_ATTRIBUTE,
        _ToolCancellationEvidence(token=_CANCELLATION_EVIDENCE_TOKEN),
    ):
        raise RuntimeError("Could not attach runtime cancellation evidence.")


def set_cancellation_tool_call_id(
    cancellation: asyncio.CancelledError,
    tool_call_id: str,
) -> None:
    if type(tool_call_id) is not str or not tool_call_id.strip():
        raise ValueError("tool_call_id must be a nonblank string.")
    _replace_cancellation_evidence(cancellation, tool_call_id=tool_call_id)


def cancellation_tool_call_id(cancellation: asyncio.CancelledError) -> str | None:
    evidence = _cancellation_evidence(cancellation)
    return None if evidence is None else evidence.tool_call_id


def set_cancellation_redactor(
    cancellation: asyncio.CancelledError,
    redactor: SecretRedactor,
) -> None:
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if redactor.has_values:
        _replace_cancellation_evidence(cancellation, redactor=redactor)


def cancellation_redactor(cancellation: asyncio.CancelledError) -> SecretRedactor | None:
    evidence = _cancellation_evidence(cancellation)
    return None if evidence is None else evidence.redactor


def set_cancellation_artifacts_by_id(
    cancellation: asyncio.CancelledError,
    artifacts_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    copied = copy_json_value(artifacts_by_id, "cancellation_artifacts_by_id")
    if type(copied) is not dict:
        raise TypeError("cancellation_artifacts_by_id must be an object.")
    _replace_cancellation_evidence(
        cancellation,
        artifacts_by_id=cast("dict[str, list[dict[str, Any]]]", copied),
    )


def cancellation_artifacts_by_id(
    cancellation: asyncio.CancelledError,
) -> dict[str, list[dict[str, Any]]] | None:
    evidence = _cancellation_evidence(cancellation)
    if evidence is None or evidence.artifacts_by_id is None:
        return None
    copied = copy_json_value(
        evidence.artifacts_by_id,
        "cancellation_artifacts_by_id",
    )
    if type(copied) is not dict:
        return None
    return cast("dict[str, list[dict[str, Any]]]", copied)


def set_cancellation_redactors_by_id(
    cancellation: asyncio.CancelledError,
    redactors_by_id: dict[str, SecretRedactor],
) -> None:
    if type(redactors_by_id) is not dict or any(
        type(tool_call_id) is not str
        or not tool_call_id.strip()
        or not isinstance(redactor, SecretRedactor)
        for tool_call_id, redactor in redactors_by_id.items()
    ):
        raise TypeError("cancellation_redactors_by_id must map tool IDs to SecretRedactor.")
    _replace_cancellation_evidence(
        cancellation,
        redactors_by_id=dict(redactors_by_id),
    )


def cancellation_redactors_by_id(
    cancellation: asyncio.CancelledError,
) -> dict[str, SecretRedactor] | None:
    evidence = _cancellation_evidence(cancellation)
    if evidence is None or evidence.redactors_by_id is None:
        return None
    return dict(evidence.redactors_by_id)


def cancellation_artifacts(
    cancellation: asyncio.CancelledError,
) -> list[dict[str, Any]]:
    if _cancellation_evidence(cancellation) is None:
        return []
    if isinstance(cancellation, RunnerCancelledError):
        return copy_json_value(cancellation.artifacts, "artifacts")
    artifacts = getattr(cancellation, "artifacts", None)
    if artifacts is not None:
        return copy_json_value(artifacts, "artifacts")
    return []


def sanitize_external_cancellation(cancellation: asyncio.CancelledError) -> None:
    """Scrub runtime-only evidence before ordinary cancellation escapes Cayu."""

    evidence = _cancellation_evidence(cancellation)
    redactors = []
    if evidence is not None:
        if evidence.redactor is not None:
            redactors.append(evidence.redactor)
        if evidence.redactors_by_id is not None:
            redactors.extend(evidence.redactors_by_id.values())
    redactor = SecretRedactor()
    for invocation_redactor in redactors:
        redactor = redactor.merged_with(invocation_redactor)
    if redactor.has_values:
        try:
            args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(cancellation, BaseException)
        except BaseException:
            args = ()
        safe_args = (
            tuple(redactor.redact_text(value) if type(value) is str else value for value in args)
            if type(args) is tuple
            else ()
        )
        _BASE_EXCEPTION_ARGS_DESCRIPTOR.__set__(cancellation, safe_args)
        namespace = _exception_namespace(cancellation)
        if namespace is not None:
            artifacts = dict.get(namespace, "artifacts")
            if artifacts is not None:
                try:
                    redacted_artifacts = redactor.redact_json(artifacts)
                except ValueError:
                    redacted_artifacts = []
                dict.__setitem__(namespace, "artifacts", redacted_artifacts)

    pop_exception_state(cancellation, _CANCELLATION_EVIDENCE_ATTRIBUTE)
    _clear_legacy_cancellation_tool_call_id(cancellation)


def _cancellation_evidence(
    cancellation: asyncio.CancelledError,
) -> _ToolCancellationEvidence | None:
    evidence = exception_state(cancellation, _CANCELLATION_EVIDENCE_ATTRIBUTE)
    if (
        type(evidence) is not _ToolCancellationEvidence
        or evidence.token is not _CANCELLATION_EVIDENCE_TOKEN
    ):
        return None
    return evidence


def _replace_cancellation_evidence(
    cancellation: asyncio.CancelledError,
    **changes: Any,
) -> None:
    evidence = _cancellation_evidence(cancellation)
    if evidence is None:
        initialize_cancellation_evidence(cancellation)
        evidence = _cancellation_evidence(cancellation)
    if evidence is None or not set_exception_state(
        cancellation,
        _CANCELLATION_EVIDENCE_ATTRIBUTE,
        replace(evidence, **changes),
    ):
        raise RuntimeError("Could not update runtime cancellation evidence.")


def _clear_legacy_cancellation_tool_call_id(cancellation: asyncio.CancelledError) -> None:
    namespace = _exception_namespace(cancellation)
    if namespace is None:
        return
    dict.pop(namespace, _LEGACY_CANCELLATION_TOOL_CALL_ID_ATTRIBUTE, None)


def _exception_namespace(error: BaseException) -> dict[str, Any] | None:
    try:
        namespace = _BASE_EXCEPTION_DICT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return None
    return namespace if type(namespace) is dict else None


def _copy_durable_proxy_secret_ref(ref: SecretRef) -> SecretRef:
    """Detach the credential identity retained for durable proxy telemetry."""

    copied = copy_secret_ref(ref)
    return SecretRef(
        name=require_clean_nonblank(
            require_durable_text(copied.name, "credential.name"),
            "credential.name",
        ),
        handle=(
            None
            if copied.handle is None
            else require_clean_nonblank(
                require_durable_text(copied.handle, "credential.handle"),
                "credential.handle",
            )
        ),
        metadata=copy_durable_json_object(copied.metadata, "credential.metadata"),
    )


def _copy_durable_proxy_authorization_result(
    result: ProxyAuthorizationResult,
) -> ProxyAuthorizationResult:
    """Detach the proxy result fields published after the tool call."""

    copied = copy_proxy_authorization_result(result)
    return ProxyAuthorizationResult(
        allowed=copied.allowed,
        reason=(
            None
            if copied.reason is None
            else require_clean_nonblank(
                require_durable_text(copied.reason, "proxy_authorization.reason"),
                "proxy_authorization.reason",
            )
        ),
        metadata=copy_durable_json_object(
            copied.metadata,
            "proxy_authorization.metadata",
        ),
    )
