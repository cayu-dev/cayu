"""Invocation-scoped secret tracking and authenticated cancellation handoff."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from cayu._exception_state import exception_state, pop_exception_state, set_exception_state
from cayu._task_wait import unexpected_child_cancellation_error
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
from cayu.runners._cleanup import pop_runner_cancellation_failure
from cayu.runtime import _runtime_records as runtime_records
from cayu.tools._redaction import InvocationRedactorSnapshot
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
_BASE_EXCEPTION_TRACEBACK_DESCRIPTOR = BaseException.__dict__["__traceback__"]
_BASE_EXCEPTION_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]
_BASE_EXCEPTION_CONTEXT_DESCRIPTOR = BaseException.__dict__["__context__"]
_BASE_EXCEPTION_DICT_DESCRIPTOR = BaseException.__dict__["__dict__"]
_RESOLUTION_CANCELLATION_DRAIN_SECONDS = 0.1
_RESOLUTION_CHILD_CANCELLATION_MESSAGE = "Invocation secret resolution cancelled by caller."

SecretResolutionScope = Literal["static", "dynamic", "unknown"]


def registered_environment_secret_resolution_scope(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Literal["static", "dynamic"]:
    """Describe an environment registration's dynamic-secret capability."""

    if registered_environment is None:
        return "static"
    # A factory may reconnect the same durable environment with a different
    # vault/proxy capability set. Without an explicit immutable capability
    # contract, factory origin is positive evidence of dynamic scope.
    if registered_environment.factory is not None or registered_environment.factory_backed:
        return "dynamic"
    environment = registered_environment.environment
    return "dynamic" if environment.vault is not None or environment.proxy is not None else "static"


def continuation_secret_resolution_scope(
    persisted_scope: SecretResolutionScope,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> SecretResolutionScope:
    """Combine durable pause evidence with the current environment capability."""

    current_scope = registered_environment_secret_resolution_scope(registered_environment)
    if persisted_scope == "dynamic" or current_scope == "dynamic":
        return "dynamic"
    if persisted_scope == "unknown":
        return "unknown"
    return "static"


def require_continuation_secret_resolution_compatibility(
    persisted_scope: SecretResolutionScope,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> None:
    """Reject a capability upgrade that would redefine already-public pause data.

    A statically classified pause may have exposed prompt or policy fields.  A
    later process must not add dynamic secret resolution to that same pending
    round, because a resolution could retroactively prove those fields secret
    after they have crossed durable and external boundaries.
    """

    if (
        persisted_scope == "static"
        and registered_environment_secret_resolution_scope(registered_environment) == "dynamic"
    ):
        raise RuntimeError(
            "Pending tool-round continuation cannot add dynamic secret resolution "
            "after static publication."
        )


@dataclass(frozen=True, slots=True)
class ProxyAuthorizationRecord:
    destination: str
    credential: SecretRef | None
    action: str | None
    metadata: dict[str, Any]
    result: ProxyAuthorizationResult


@dataclass(frozen=True, slots=True)
class InvocationPublicationSnapshot:
    """Final secret scope and whether invocation output is safe to publish."""

    redactor: SecretRedactor
    unsafe_output: bool
    secret_scope_incomplete: bool = False


class InvocationSecretTracker:
    """Own the secret-redaction scope for one tool invocation."""

    def __init__(self, redactor: SecretRedactor) -> None:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        self._redactor = redactor
        self._revision = 0
        self._active_resolutions: set[object] = set()
        self._secret_scope_uncertain = False
        self._ambiguous_capture_revisions: list[int] = []
        self._sealed = False
        self._publication_snapshot: InvocationPublicationSnapshot | None = None

    @property
    def redactor(self) -> SecretRedactor:
        return self._redactor

    def snapshot(self) -> InvocationRedactorSnapshot:
        """Return an atomic view for an external capture boundary."""

        return InvocationRedactorSnapshot(
            revision=self._revision,
            redactor=self._redactor,
        )

    def begin_resolution(self) -> object:
        """Claim one secret resolution before crossing its asynchronous boundary."""

        if self._sealed:
            raise RuntimeError("Secret resolution is unavailable after tool publication.")
        token = object()
        self._active_resolutions.add(token)
        return token

    def complete_resolution(self, token: object, secret: ResolvedSecret) -> bool:
        """Record a secret, or reject it without raising through a secret-bearing frame."""

        if token not in self._active_resolutions:
            return False
        self._active_resolutions.remove(token)
        if self._sealed:
            return False
        self.record(secret)
        return True

    def abandon_resolution(
        self,
        token: object,
        *,
        secret_scope_uncertain: bool = False,
    ) -> bool:
        """Revoke a pending resolution claim and report whether it was still active."""

        if token not in self._active_resolutions:
            return False
        self._active_resolutions.remove(token)
        if secret_scope_uncertain:
            # A dispatched opaque resolver may still return after caller
            # cancellation. Its result is fenced from mutating this tracker,
            # but absence of that unknown value can never prove publication
            # safe for the remainder of the invocation.
            self._secret_scope_uncertain = True
        return True

    def record(self, secret: ResolvedSecret) -> None:
        if type(secret) is not ResolvedSecret:
            raise TypeError("Resolved secrets must be ResolvedSecret instances.")
        if self._sealed:
            raise RuntimeError("Secret registration is unavailable after tool publication.")
        updated = self._redactor.with_secret(secret)
        if updated is self._redactor:
            return
        self._redactor = updated
        self._revision += 1

    def record_ambiguous_output_capture(self, revision: int) -> None:
        """Remember a bounded output whose safety depends on this exact revision."""

        if type(revision) is not int or revision < 0:
            raise TypeError("Output capture revision must be a non-negative integer.")
        if self._sealed:
            raise RuntimeError("Output capture completed after tool publication.")
        self._ambiguous_capture_revisions.append(revision)

    def record_ambiguous_runner_capture(self, revision: int) -> None:
        """Compatibility alias for runner-specific capture callers."""

        self.record_ambiguous_output_capture(revision)

    def seal_for_publication(self) -> InvocationPublicationSnapshot:
        """Freeze secret registration and classify revision-stale bounded captures."""

        if self._publication_snapshot is not None:
            return self._publication_snapshot
        self._sealed = True
        secret_scope_incomplete = bool(self._active_resolutions) or self._secret_scope_uncertain
        unsafe_output = secret_scope_incomplete or any(
            revision != self._revision for revision in self._ambiguous_capture_revisions
        )
        snapshot = InvocationPublicationSnapshot(
            redactor=self._redactor,
            unsafe_output=unsafe_output,
            secret_scope_incomplete=secret_scope_incomplete,
        )
        self._publication_snapshot = snapshot
        return snapshot


async def _ignore_redactor_change(_snapshot: InvocationRedactorSnapshot) -> None:
    return None


async def _await_resolved_secret_operation(
    operation: asyncio.Task[ResolvedSecret],
    *,
    abandon_resolution: Callable[[], bool],
) -> ResolvedSecret:
    """Return a persisted resolution or fence a cancelled remote lookup."""

    current_task = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    observed_cancelling = 0 if current_task is None else current_task.cancelling()
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError as exc:
            current_cancelling = 0 if current_task is None else current_task.cancelling()
            if current_cancelling <= observed_cancelling:
                if not operation.done():
                    raise
                break
            if cancellation is None:
                cancellation = exc
            observed_cancelling = current_cancelling
            if abandon_resolution():
                # The remote lookup had not yet registered its secret or
                # started the durable redactor projection. Revoke its claim
                # before requesting cancellation so an opaque adapter that
                # keeps running cannot mutate invocation publication later.
                operation.cancel(_RESOLUTION_CHILD_CANCELLATION_MESSAGE)
                deadline = (
                    asyncio.get_running_loop().time() + _RESOLUTION_CANCELLATION_DRAIN_SECONDS
                )
                while not operation.done():
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(operation),
                            timeout=remaining,
                        )
                    except asyncio.CancelledError as later:
                        current_cancelling = (
                            0 if current_task is None else current_task.cancelling()
                        )
                        if current_cancelling <= observed_cancelling:
                            if not operation.done():
                                raise
                            break
                        observed_cancelling = current_cancelling
                        if cancellation is None:  # pragma: no cover - construction invariant
                            cancellation = later
                    except TimeoutError:
                        break
                    except BaseException:
                        if not operation.done():
                            raise
                        break
                if operation.done():
                    _consume_resolution_task(operation)
                else:
                    operation.add_done_callback(_consume_resolution_task)
                del operation, abandon_resolution
                raise cancellation from None
            # Resolution has already registered its redactor and entered the
            # durable projection boundary. Settle that write before returning
            # cancellation; abandoning an in-flight store mutation would make
            # its eventual commit race with terminal publication.
            abandon_resolution = _resolution_already_accepted
    # Do not let a cancellation arriving in the same loop turn as child
    # completion escape after a successful secret has been returned.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        current_cancelling = 0 if current_task is None else current_task.cancelling()
        if current_cancelling <= observed_cancelling:
            raise
        if cancellation is None:
            cancellation = exc
        observed_cancelling = current_cancelling
    try:
        result = operation.result()
        error: BaseException | None = None
    except BaseException as exc:
        result = None
        error = exc
    del operation
    if isinstance(error, asyncio.CancelledError) and cancellation is None:
        error = unexpected_child_cancellation_error(
            error,
            operation="Invocation secret resolution publication",
        )
    if cancellation is not None:
        result = None
        error = None
        raise cancellation from None
    if error is not None:
        raise error
    if type(result) is not ResolvedSecret:
        raise RuntimeError("Secret resolution publication returned no resolved secret.")
    return result


def _resolution_already_accepted() -> bool:
    return False


def _consume_resolution_task(task: asyncio.Task[ResolvedSecret]) -> None:
    """Observe a detached resolver outcome without retaining its secret or error."""

    with suppress(BaseException):
        task.result()


async def _dispatch_tracked_secret_resolution(
    tracker: InvocationSecretTracker,
    operation_factory: Callable[[object], Coroutine[Any, Any, ResolvedSecret]],
) -> ResolvedSecret:
    """Checkpoint cancellation, claim ownership, and dispatch one resolver."""

    # Pending cancellation is authoritative before dispatch. In particular,
    # do not create an adapter task or an active publication claim first.
    await asyncio.sleep(0)
    token = tracker.begin_resolution()
    operation: asyncio.Task[ResolvedSecret] | None = None
    try:
        operation = asyncio.create_task(operation_factory(token))
    except BaseException:
        tracker.abandon_resolution(token)
        raise
    finally:
        del operation_factory

    def abandon(
        tracker: InvocationSecretTracker = tracker,
        token: object = token,
    ) -> bool:
        return tracker.abandon_resolution(token, secret_scope_uncertain=True)

    del tracker, token
    return await _await_resolved_secret_operation(
        operation,
        abandon_resolution=abandon,
    )


class _TrackingVault(Vault):
    """Track secrets resolved directly through a tool's vault boundary."""

    def __init__(
        self,
        vault: Vault,
        tracker: InvocationSecretTracker,
        on_redactor_change: Callable[[InvocationRedactorSnapshot], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(vault, Vault):
            raise TypeError("vault must be a Vault.")
        if not isinstance(tracker, InvocationSecretTracker):
            raise TypeError("tracker must be an InvocationSecretTracker.")
        if on_redactor_change is not None and not callable(on_redactor_change):
            raise TypeError("on_redactor_change must be callable.")
        self._vault = vault
        self._tracker = tracker
        self._on_redactor_change = on_redactor_change or _ignore_redactor_change

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

        def operation_factory(token: object) -> Coroutine[Any, Any, ResolvedSecret]:
            return self._resolve_and_publish(
                copied_ref,
                scope=copied_scope,
                token=token,
            )

        return await _dispatch_tracked_secret_resolution(
            self._tracker,
            operation_factory,
        )

    async def _resolve_and_publish(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None,
        token: object,
    ) -> ResolvedSecret:
        secret: ResolvedSecret | None = None
        copied_secret: ResolvedSecret | None = None
        try:
            secret = await self._vault.resolve(
                ref,
                scope=None if scope is None else copy_json_object(scope, "scope"),
            )
        except BaseException:
            self._tracker.abandon_resolution(token)
            raise
        if type(secret) is not ResolvedSecret:
            self._tracker.abandon_resolution(token)
            raise TypeError("Vault secret resolution must return ResolvedSecret.")
        try:
            copied_secret = copy_resolved_secret(secret)
            secret = None
            accepted = self._tracker.complete_resolution(token, copied_secret)
            if not accepted:
                raise RuntimeError("Secret resolution completed after tool publication.")
            await self._on_redactor_change(self._tracker.snapshot())
            return copy_resolved_secret(copied_secret)
        except BaseException:
            secret = None
            copied_secret = None
            raise


class _TrackingCredentialProxy(CredentialProxy):
    def __init__(
        self,
        proxy: CredentialProxy,
        tracker: InvocationSecretTracker,
        on_authorize: Callable[[ProxyAuthorizationRecord], None],
        on_redactor_change: Callable[[InvocationRedactorSnapshot], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(proxy, CredentialProxy):
            raise TypeError("proxy must be a CredentialProxy.")
        if not isinstance(tracker, InvocationSecretTracker):
            raise TypeError("tracker must be an InvocationSecretTracker.")
        if not callable(on_authorize):
            raise TypeError("on_authorize must be callable.")
        if on_redactor_change is not None and not callable(on_redactor_change):
            raise TypeError("on_redactor_change must be callable.")
        self._proxy = proxy
        self._tracker = tracker
        self._on_authorize = on_authorize
        self._on_redactor_change = on_redactor_change or _ignore_redactor_change

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        copied_ref = copy_secret_ref(ref)
        copied_scope = None if scope is None else copy_json_object(scope, "scope")

        def operation_factory(token: object) -> Coroutine[Any, Any, ResolvedSecret]:
            return self._resolve_and_publish(
                copied_ref,
                scope=copied_scope,
                token=token,
            )

        return await _dispatch_tracked_secret_resolution(
            self._tracker,
            operation_factory,
        )

    async def _resolve_and_publish(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None,
        token: object,
    ) -> ResolvedSecret:
        secret: ResolvedSecret | None = None
        copied_secret: ResolvedSecret | None = None
        try:
            secret = await self._proxy.resolve(
                ref,
                scope=None if scope is None else copy_json_object(scope, "scope"),
            )
        except BaseException:
            self._tracker.abandon_resolution(token)
            raise
        if type(secret) is not ResolvedSecret:
            self._tracker.abandon_resolution(token)
            raise TypeError("Proxy secret resolution must return ResolvedSecret.")
        try:
            copied_secret = copy_resolved_secret(secret)
            secret = None
            accepted = self._tracker.complete_resolution(token, copied_secret)
            if not accepted:
                raise RuntimeError("Secret resolution completed after tool publication.")
            await self._on_redactor_change(self._tracker.snapshot())
            return copy_resolved_secret(copied_secret)
        except BaseException:
            secret = None
            copied_secret = None
            raise

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
    tracker: InvocationSecretTracker,
    on_redactor_change: Callable[[InvocationRedactorSnapshot], Awaitable[None]],
) -> Vault | None:
    if registered_environment is None:
        return None
    vault = registered_environment.environment.vault
    if vault is None:
        return None
    return _TrackingVault(vault, tracker, on_redactor_change)


def proxy_for_environment(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    tracker: InvocationSecretTracker,
    on_authorize: Callable[[ProxyAuthorizationRecord], None],
    on_redactor_change: Callable[[InvocationRedactorSnapshot], Awaitable[None]],
) -> CredentialProxy | None:
    if registered_environment is None:
        return None
    proxy = registered_environment.environment.proxy
    if proxy is None:
        return None
    return _TrackingCredentialProxy(proxy, tracker, on_authorize, on_redactor_change)


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


def has_cancellation_evidence(cancellation: asyncio.CancelledError) -> bool:
    """Return whether the runtime authenticated this cancellation handoff."""

    return _cancellation_evidence(cancellation) is not None


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

    runner_failure = pop_runner_cancellation_failure(cancellation)
    if runner_failure is not None:
        try:
            # Import at the publication boundary to avoid making invocation
            # secret tracking depend on the built-in tools package at import
            # time. The authenticated object was visible to extension code, so
            # rebuild it now rather than trusting its current mutable state.
            from cayu.tools._runner import sanitize_runner_failure

            runner_failure = sanitize_runner_failure(runner_failure)
        except BaseException:
            runner_failure = None
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
    # A tool can await its runner while handling a secret-bearing exception.
    # Python attaches that active exception only when cancellation crosses back
    # into the tool, after the runner boundary has already detached its own
    # frames. This is the first runtime-owned boundary that sees the complete
    # tool failure tree, so sever every historical exception path here before
    # ordinary cancellation escapes Cayu.
    _BASE_EXCEPTION_TRACEBACK_DESCRIPTOR.__set__(cancellation, None)
    _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__set__(cancellation, None)
    _BASE_EXCEPTION_CONTEXT_DESCRIPTOR.__set__(cancellation, None)
    if runner_failure is not None:
        _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__set__(cancellation, runner_failure)


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
