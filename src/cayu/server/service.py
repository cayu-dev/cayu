"""Secure-by-default product service composition for Cayu applications.

The service owns a deliberately small product API and keeps Cayu's raw control
plane behind a separate operator policy. Application-owned storage remains the
authority for tenant/resource ownership; Cayu session and task identifiers are
never accepted from or projected to product callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Never, Protocol, TypeGuard, cast, runtime_checkable
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import InitErrorDetails
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import EventType
from cayu.core.messages import Message
from cayu.runtime.app import CayuApp
from cayu.runtime.service_manifest import (
    PublicServiceManifest,
    RuntimeStoreDurability,
    ServiceIdentityStoreKind,
    ServiceMode,
)
from cayu.runtime.sessions import RunRequest, SessionStatus
from cayu.runtime.tasks import Task, TaskCreate, TaskStatus
from cayu.server.config import (
    AuthenticatedAccess,
    OpenAccess,
    ServerAccessConfig,
    _raise_redacted_config_error,
    _validate_with_redacted_errors,
    normalize_api_path,
)
from cayu.vaults import (
    REDACTED_SECRET,
    SecretRedactionCapacityError,
    SecretRedactionStream,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send
    from starlette.types import Message as ASGIMessage

MAX_PUBLIC_RESULT_CHARS = 100_000
_MAX_PUBLIC_RESULT_UTF8_BYTES = MAX_PUBLIC_RESULT_CHARS * 4
_PUBLIC_RESULT_CAPTURE_BYTES = _MAX_PUBLIC_RESULT_UTF8_BYTES + len(REDACTED_SECRET.encode("utf-8"))
_PUBLIC_RESULT_SOURCE_CHARS_PER_CHUNK = 4096
MAX_PRODUCT_IDENTITY_CHARS = 512
MAX_PRODUCT_REQUEST_BYTES = 1024 * 1024
PRODUCT_EXECUTION_LEASE_SECONDS = 120
PRODUCT_EXECUTION_HEARTBEAT_SECONDS = 30
PRODUCT_EXECUTION_HEARTBEAT_TIMEOUT_SECONDS = 10

_INVALID_PRODUCT_REQUEST_DETAIL = "Invalid product request."
_OVERSIZED_PRODUCT_REQUEST_DETAIL = "Product request exceeds the server byte limit."


class _ProductWorkReconciliationRequired(RuntimeError):
    """Durable Cayu work may exist and must not be terminalized or redispatched."""


def _product_request_fingerprint(*, agent_name: str, request_text: str) -> str:
    fingerprint_input = json.dumps(
        {"agent_name": agent_name, "request": request_text, "schema_version": 1},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(fingerprint_input).hexdigest()


def _private_product_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"Cache-Control": "private, no-store"},
    )


def _parse_product_json_without_duplicate_keys(body: bytes) -> None:
    """Validate one bounded JSON body without retaining attacker-controlled values."""

    def reject_constant(value: str) -> Never:
        raise ValueError(f"Non-finite JSON number {value!r} is not supported.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object keys are not supported.")
            result[key] = value
        return result

    json.loads(
        body,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


class _ProductBoundaryMiddleware:
    """Bound product JSON before parsing and make product responses non-cacheable."""

    def __init__(self, app: ASGIApp, *, product_operations_path: str) -> None:
        self.app = app
        self.product_operations_path = product_operations_path
        product_api_path, _separator, _operations = product_operations_path.rpartition("/")
        self.product_api_path = product_api_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path != self.product_api_path and not path.startswith(f"{self.product_api_path}/"):
            await self.app(scope, receive, send)
            return

        if scope.get("method") == "POST" and path == self.product_operations_path:
            body, error_response = await _bounded_product_json_body(scope, receive)
            if error_response is not None:
                await error_response(scope, receive, send)
                return
            if body is None:
                return
            replayed = False

            async def replay_receive() -> ASGIMessage:
                nonlocal replayed
                if replayed:
                    return {"type": "http.disconnect"}
                replayed = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }

            receive = replay_receive

        async def send_no_store(message: ASGIMessage) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "private, no-store"
            await send(message)

        await self.app(scope, receive, send_no_store)


async def _bounded_product_json_body(
    scope: Scope,
    receive: Receive,
) -> tuple[bytes | None, JSONResponse | None]:
    declared_lengths = [
        value for key, value in scope.get("headers", ()) if key.lower() == b"content-length"
    ]
    if len(declared_lengths) > 1:
        return None, _private_product_error_response(
            400,
            _INVALID_PRODUCT_REQUEST_DETAIL,
        )
    if declared_lengths:
        try:
            declared_bytes = int(declared_lengths[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return None, _private_product_error_response(
                400,
                _INVALID_PRODUCT_REQUEST_DETAIL,
            )
        if declared_bytes < 0:
            return None, _private_product_error_response(
                400,
                _INVALID_PRODUCT_REQUEST_DETAIL,
            )
        if declared_bytes > MAX_PRODUCT_REQUEST_BYTES:
            return None, _private_product_error_response(
                413,
                _OVERSIZED_PRODUCT_REQUEST_DETAIL,
            )

    received = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None, None
        if message["type"] != "http.request":
            return None, _private_product_error_response(
                400,
                _INVALID_PRODUCT_REQUEST_DETAIL,
            )
        chunk = message.get("body", b"")
        if len(received) + len(chunk) > MAX_PRODUCT_REQUEST_BYTES:
            return None, _private_product_error_response(
                413,
                _OVERSIZED_PRODUCT_REQUEST_DETAIL,
            )
        received.extend(chunk)
        if not message.get("more_body", False):
            break

    try:
        _parse_product_json_without_duplicate_keys(bytes(received))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, _private_product_error_response(
            400,
            _INVALID_PRODUCT_REQUEST_DETAIL,
        )
    return bytes(received), None


class ProductPrincipal(BaseModel):
    """Trusted product identity resolved by server-side authentication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    subject_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)

    @field_validator("tenant_id", "subject_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


ProductAuthResult = ProductPrincipal | Mapping[str, str]
ProductAuthDependency = Callable[
    [Request],
    ProductAuthResult | Awaitable[ProductAuthResult],
]


class _ProductAccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    dependency: Any = Field(exclude=True, repr=False)

    @model_validator(mode="wrap")
    @classmethod
    def validate_dependency(cls, value: Any, handler: Any) -> _ProductAccess:
        if isinstance(value, Mapping) and "dependency" in value:
            dependency = value["dependency"]
            if not callable(dependency):
                _raise_redacted_config_error(
                    cls.__name__,
                    [
                        InitErrorDetails(
                            type="value_error",
                            loc=("dependency",),
                            input=None,
                            ctx={
                                "error": ValueError(
                                    "Product access requires a callable authentication dependency."
                                )
                            },
                        )
                    ],
                )
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid product-access configuration.",
        )


class AuthenticatedProductAccess(_ProductAccess):
    """Production product authentication and trusted tenant resolution."""

    kind: Literal["authenticated"] = "authenticated"


class DevelopmentProductAccess(_ProductAccess):
    """Explicit local-development product identity adapter."""

    kind: Literal["development"] = "development"


class PlaceholderProductAccess(BaseModel):
    """Fail-closed missing product authentication surfaced by deployment checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["placeholder"] = "placeholder"

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> PlaceholderProductAccess:
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid placeholder product-access configuration.",
        )


ProductAccess = AuthenticatedProductAccess | DevelopmentProductAccess | PlaceholderProductAccess


class PlaceholderOperatorAccess(BaseModel):
    """Fail-closed missing operator authentication surfaced by deployment checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["placeholder"] = "placeholder"

    @model_validator(mode="wrap")
    @classmethod
    def redact_validation_inputs(cls, value: Any, handler: Any) -> PlaceholderOperatorAccess:
        return _validate_with_redacted_errors(
            cls.__name__,
            value,
            handler,
            message="Invalid placeholder operator-access configuration.",
        )


OperatorAccess = ServerAccessConfig | PlaceholderOperatorAccess


class ProductOperation(BaseModel):
    """Application-owned binding between public identity and private Cayu authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    public_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    work_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    idempotency_key: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    request_fingerprint: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    session_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    task_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    request_text: str = Field(max_length=100_000)
    status: Literal["pending", "completed", "failed"]
    result: str | None = Field(max_length=100_000)

    @field_validator(
        "tenant_id",
        "public_id",
        "work_id",
        "idempotency_key",
        "request_fingerprint",
        "session_id",
        "task_id",
    )
    @classmethod
    def validate_clean_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("request_text")
    @classmethod
    def validate_request_text(cls, value: str) -> str:
        return require_durable_nonblank(value, "request_text")


class ProductOperationReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation: ProductOperation
    created: bool


class ProductOperationExecutionClaim(BaseModel):
    """One authoritative attempt to execute pending product work."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation: ProductOperation
    acquired: bool


def _validated_product_store_operation(
    value: object,
    *,
    source: str,
    expected: Mapping[str, object],
    validate_complete: bool,
) -> ProductOperation:
    """Snapshot declared fields and validate only what the caller will consume."""

    if not isinstance(value, ProductOperation):
        raise TypeError(f"Product store must return ProductOperation from {source}().")
    try:
        fields = {
            field_name: getattr(value, field_name) for field_name in ProductOperation.model_fields
        }
        operation = (
            ProductOperation.model_validate(fields)
            if validate_complete
            else ProductOperation.model_construct(**fields)
        )
    except (AttributeError, TypeError, ValueError):
        raise TypeError(f"Product store returned an invalid operation from {source}().") from None
    if any(
        type(getattr(operation, field_name)) is not type(expected_value)
        or getattr(operation, field_name) != expected_value
        for field_name, expected_value in expected.items()
    ):
        raise RuntimeError(f"Product store returned inconsistent authority from {source}().")
    return operation


class ProductIdempotencyConflict(Exception):
    """An idempotency identity is already bound to different trusted work."""


class ProductExecutionClaimLost(Exception):
    """A product worker no longer owns the operation it attempted to settle."""


class ProductOperationSettlementConflict(Exception):
    """A product operation already has a different terminal result."""


@runtime_checkable
class ProductOperationStore(Protocol):
    """Application-owned authorization, idempotency, and execution boundary.

    Claims, heartbeats, and terminal writes must be atomic in the backing store
    and use its clock for lease expiry. Concurrent heartbeats for the same claim
    must not shorten its lease. Terminal rows retain the settling ``claim_id``;
    repeating a write with that identity reconstructs the committed result after
    an ambiguous acknowledgement loss.
    """

    category: ServiceIdentityStoreKind

    async def reserve(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation: ...

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None: ...

    async def claim_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        """Acquire, renew, or observe one pending operation atomically."""
        ...

    async def heartbeat_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        """Extend pending ownership or recognize this claim's terminal write.

        Implementations must return true when the operation is still pending
        under ``claim_id`` and the lease was renewed, or when that exact claim
        already settled the operation. False means another execution owns or
        settled the work.
        """
        ...

    async def release_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
    ) -> bool:
        """Relinquish one pending execution claim without clearing a successor.

        Implementations must atomically clear ``claim_id`` only while that exact
        claim still owns pending work. True means the claim was cleared or was
        already unclaimed. False means the operation is terminal or another claim
        owns it, so this caller no longer owns the operation in either case.
        Repeating the call after an ambiguous acknowledgement must be safe.
        """
        ...

    async def finish(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: Literal["completed", "failed"],
        result: str | None,
    ) -> ProductOperation:
        """Conditionally settle owned work or reconstruct the same terminal write."""
        ...


class CreateOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    request: str = Field(min_length=1, max_length=100_000)

    @field_validator("request")
    @classmethod
    def validate_request(cls, value: str) -> str:
        return require_durable_nonblank(value, "request")


class PublicOperationResponse(BaseModel):
    """The complete allow-list for customer-visible operation state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    status: Literal["pending", "completed", "failed"]
    result: str | None = Field(max_length=100_000)


class CayuService:
    """One assembled and inspectable maintained Cayu product service."""

    __slots__ = (
        "_agent_name",
        "_asgi_app",
        "_assembly_token",
        "_cayu_app",
        "_manifest",
        "_product_store",
        "_route_signature",
    )

    def __init__(
        self,
        *,
        cayu_app: CayuApp,
        asgi_app: FastAPI,
        manifest: PublicServiceManifest,
        product_store: ProductOperationStore,
        agent_name: str,
        _assembly_token: object | None = None,
    ) -> None:
        if _assembly_token is not _SERVICE_ASSEMBLY_TOKEN:
            raise TypeError("CayuService instances are created only by create_agent_service().")
        self._cayu_app = cayu_app
        self._asgi_app = asgi_app
        self._manifest = manifest
        self._product_store = product_store
        self._agent_name = agent_name
        self._assembly_token = _assembly_token
        self._route_signature: tuple[object, ...] | None = None

    @property
    def cayu_app(self) -> CayuApp:
        return self._cayu_app

    @property
    def asgi_app(self) -> FastAPI:
        return self._asgi_app

    @property
    def manifest(self) -> PublicServiceManifest:
        return self._manifest

    @property
    def product_store(self) -> ProductOperationStore:
        return self._product_store

    @property
    def agent_name(self) -> str:
        return self._agent_name

    async def execute_work(self, work_id: str) -> ProductOperation | None:
        """Reload trusted ownership from product storage, then run opaque queued work."""

        work_id = require_durable_clean_nonblank(work_id, "work_id")
        if len(work_id) > MAX_PRODUCT_IDENTITY_CHARS:
            raise ValueError(f"work_id must not exceed {MAX_PRODUCT_IDENTITY_CHARS} characters.")
        claim_id = f"claim_{uuid4().hex}"
        claim = await _reconcile_ambiguous_product_store_write(
            lambda: self.product_store.claim_execution(
                work_id=work_id,
                claim_id=claim_id,
                lease_seconds=PRODUCT_EXECUTION_LEASE_SECONDS,
            ),
            operation="product execution claim",
        )
        if claim is None:
            return None
        if not isinstance(claim, ProductOperationExecutionClaim):
            raise TypeError(
                "Product store must return ProductOperationExecutionClaim from claim_execution()."
            )
        if type(claim.acquired) is not bool:
            raise TypeError("Product store returned an invalid execution claim.")
        operation = _validated_product_store_operation(
            claim.operation,
            source="claim_execution",
            expected={"work_id": work_id},
            validate_complete=True,
        )
        if not claim.acquired:
            return operation
        if operation.status != "pending":
            raise RuntimeError("Product store acquired terminal product work.")

        async def execute_and_settle() -> ProductOperation:
            try:
                if operation.request_fingerprint != _product_request_fingerprint(
                    agent_name=self.agent_name,
                    request_text=operation.request_text,
                ):
                    raise _ProductWorkReconciliationRequired(
                        "Product work does not match this service's agent and request contract."
                    )
                task_is_ready = await _create_or_verify_product_task(
                    service=self,
                    operation=operation,
                )
            except _ProductWorkReconciliationRequired as reconciliation_error:
                try:
                    await _release_product_execution_claim_resisting_cancellation(
                        self.product_store,
                        work_id=operation.work_id,
                        claim_id=claim_id,
                    )
                except asyncio.CancelledError as release_cancellation:
                    _raise_cancellation_with_failure(
                        release_cancellation,
                        reconciliation_error,
                        operation="product operation reconciliation handoff",
                    )
                except BaseException as release_error:
                    raise BaseExceptionGroup(
                        "Product work reconciliation and execution-claim release both failed.",
                        [reconciliation_error, release_error],
                    ) from None
                raise
            except (BaseExceptionGroup, Exception, asyncio.CancelledError) as execution_error:
                await _terminalize_failed_product_operation(
                    self.product_store,
                    operation,
                    claim_id,
                    execution_error,
                )
            if not task_is_ready:
                # Once Cayu work has advanced beyond the initial pending task,
                # reconstructing its public result belongs to the application's
                # worker-recovery contract. Preserve the honest pending product
                # state without redispatching provider work or falsely failing it.
                await _release_product_execution_claim_resisting_cancellation(
                    self.product_store,
                    work_id=operation.work_id,
                    claim_id=claim_id,
                )
                return operation
            try:
                await _retain_product_execution_claim(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                )
                outcome_status, final_text = await _run_product_operation_bounded(
                    self.cayu_app,
                    RunRequest(
                        agent_name=self.agent_name,
                        messages=[Message.text("user", operation.request_text)],
                        session_id=operation.session_id,
                        task_id=operation.task_id,
                    ),
                )
            except (BaseExceptionGroup, Exception, asyncio.CancelledError) as execution_error:
                await _terminalize_failed_product_operation(
                    self.product_store,
                    operation,
                    claim_id,
                    execution_error,
                )
            return await _finish_product_operation_resisting_cancellation(
                self.product_store,
                work_id=operation.work_id,
                claim_id=claim_id,
                status=("completed" if outcome_status is SessionStatus.COMPLETED else "failed"),
                result=(final_text if outcome_status is SessionStatus.COMPLETED else None),
            )

        return await _run_product_operation_with_heartbeat(
            service=self,
            operation=operation,
            claim_id=claim_id,
            execute_and_settle=execute_and_settle,
        )


async def _run_product_operation_bounded(
    app: CayuApp,
    request: RunRequest,
) -> tuple[SessionStatus, str]:
    """Consume one run without retaining its complete event or delta history."""

    status = SessionStatus.INTERRUPTED
    terminal_observed = False
    current_turn_text = _BoundedProductResultCapture(app)
    final_text = ""
    try:
        async for event in app.run(request):
            payload = event.payload or {}
            if event.type == EventType.MODEL_STARTED:
                current_turn_text.abort()
                current_turn_text = _BoundedProductResultCapture(app)
                final_text = ""
            elif event.type == EventType.MODEL_TEXT_DELTA:
                delta = payload.get("delta")
                if isinstance(delta, str):
                    current_turn_text.append(delta)
            elif event.type == EventType.MODEL_COMPLETED:
                captured_text = current_turn_text.finish_complete()
                final_text = captured_text if captured_text is not None else ""
            elif event.type == EventType.SESSION_COMPLETED:
                status = SessionStatus.COMPLETED
                terminal_observed = True
            elif event.type == EventType.SESSION_FAILED:
                status = SessionStatus.FAILED
                terminal_observed = True
            elif event.type == EventType.SESSION_INTERRUPTED:
                status = SessionStatus.INTERRUPTED
                terminal_observed = True
    except Exception:
        if not terminal_observed:
            status = SessionStatus.FAILED
    finally:
        current_turn_text.abort()
    if current_turn_text.failed and status is SessionStatus.COMPLETED:
        status = SessionStatus.FAILED
    return status, final_text


class _BoundedProductResultCapture:
    """Retain one marker-safe result prefix after cross-delta secret redaction."""

    def __init__(self, app: CayuApp) -> None:
        self._stream: SecretRedactionStream = app.stream_redacted_bytes(
            max_retained_bytes=_PUBLIC_RESULT_CAPTURE_BYTES,
        )
        self._content = bytearray()
        self._released_bytes = 0
        self._prefix_finalized = False
        self._finished = False
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def append(self, chunk: str) -> None:
        if self._finished or self._prefix_finalized:
            return
        for offset in range(0, len(chunk), _PUBLIC_RESULT_SOURCE_CHARS_PER_CHUNK):
            try:
                self._capture(
                    self._stream.feed(
                        chunk[offset : offset + _PUBLIC_RESULT_SOURCE_CHARS_PER_CHUNK].encode(
                            "utf-8", "replace"
                        )
                    )
                )
            except SecretRedactionCapacityError:
                self._failed = True
                self._stream.abort()
                return
            if self._prefix_finalized:
                self._stream.abort()
                break

    def finish_complete(self) -> str | None:
        if self._finished:
            raise RuntimeError("Product result capture is already finished.")
        self._finished = True
        if self._failed:
            return None
        if not self._prefix_finalized:
            try:
                self._capture(self._stream.finish_complete())
            except SecretRedactionCapacityError:
                self._failed = True
                return None
        return _bounded_public_result_text(bytes(self._content).decode("utf-8", "ignore"))

    def abort(self) -> None:
        if self._finished:
            return
        self._stream.abort()
        self._finished = True

    def _capture(self, released: bytes) -> None:
        if not released:
            return
        self._released_bytes += len(released)
        remaining = _PUBLIC_RESULT_CAPTURE_BYTES - len(self._content)
        if remaining > 0:
            self._content.extend(released[:remaining])
        if self._released_bytes >= _PUBLIC_RESULT_CAPTURE_BYTES:
            self._prefix_finalized = True


def _bounded_public_result_text(value: str) -> str:
    """Apply the character limit without publishing a partial redaction marker."""

    if len(value) <= MAX_PUBLIC_RESULT_CHARS:
        return value
    marker_start = value.rfind(
        REDACTED_SECRET,
        0,
        MAX_PUBLIC_RESULT_CHARS + len(REDACTED_SECRET),
    )
    if marker_start >= 0 and marker_start < MAX_PUBLIC_RESULT_CHARS < marker_start + len(
        REDACTED_SECRET
    ):
        return value[:marker_start]
    return value[:MAX_PUBLIC_RESULT_CHARS]


async def _create_or_verify_product_task(
    *,
    service: CayuService,
    operation: ProductOperation,
) -> bool:
    """Create initial Cayu work or verify its exact durable replay state.

    True means the task is still the original unstarted task and may enter a new
    run. False means durable Cayu work has already advanced and must not be
    redispatched by this bounded product-service contract.
    """

    task_store = service.cayu_app.task_store
    if task_store is None:
        raise RuntimeError("task_store is required to execute product work.")
    request = TaskCreate(
        task_id=operation.task_id,
        type="public_agent_operation",
        session_id=operation.session_id,
        assigned_agent_name=service.agent_name,
    )
    try:
        task = await service.cayu_app.create_task(request)
    except Exception as creation_error:
        try:
            task = await task_store.load_task(operation.task_id)
        except Exception as reconciliation_error:
            combined = BaseExceptionGroup(
                "Product task creation and acknowledgement reconciliation failed.",
                [creation_error, reconciliation_error],
            )
            raise _ProductWorkReconciliationRequired(
                "Product task creation acknowledgement could not be reconciled safely."
            ) from combined
        if task is None:
            raise creation_error

    if not _product_task_is_safe_to_start(
        task,
        operation=operation,
        agent_name=service.agent_name,
    ):
        return False
    try:
        session_state = await service.cayu_app.session_store.load_state(operation.session_id)
    except Exception as load_error:
        raise _ProductWorkReconciliationRequired(
            "Product session state could not be reconciled safely."
        ) from load_error
    return session_state is None


def _product_task_is_safe_to_start(
    task: object,
    *,
    operation: ProductOperation,
    agent_name: str,
) -> bool:
    """Validate stable task authority and classify its initial lifecycle state."""

    if not isinstance(task, Task):
        raise TypeError("Task store returned an invalid product task.")
    try:
        task = Task.model_validate(
            {field_name: getattr(task, field_name) for field_name in Task.model_fields}
        )
    except (AttributeError, TypeError, ValueError):
        raise TypeError("Task store returned an invalid product task.") from None
    if (
        task.id != operation.task_id
        or task.type != "public_agent_operation"
        or task.session_id != operation.session_id
        or task.parent_task_id is not None
        or task.assigned_agent_name != agent_name
    ):
        raise RuntimeError("Existing Cayu task conflicts with product work authority.")
    if task.status != TaskStatus.PENDING:
        return False
    return not (
        task.title is not None
        or task.description is not None
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.status_reason is not None
        or task.status_payload is not None
        or task.input
        or task.result is not None
        or task.error is not None
        or task.metadata
        or task.started_at is not None
        or task.completed_at is not None
    )


async def _run_product_operation_with_heartbeat(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    execute_and_settle: Callable[[], Awaitable[ProductOperation]],
) -> ProductOperation:
    """Retain authoritative ownership through runtime work and terminal settlement."""

    stop_heartbeat = asyncio.Event()

    async def execute() -> ProductOperation:
        try:
            return await execute_and_settle()
        finally:
            stop_heartbeat.set()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop_heartbeat.wait(),
                    timeout=PRODUCT_EXECUTION_HEARTBEAT_SECONDS,
                )
                return
            except TimeoutError:
                await _retain_product_execution_claim(
                    service=service,
                    operation=operation,
                    claim_id=claim_id,
                )

    execution_task = asyncio.create_task(
        execute(),
        name="cayu-public-operation-execution",
    )
    heartbeat_task = asyncio.create_task(
        heartbeat(),
        name="cayu-public-operation-heartbeat",
    )
    caller_cancellation: asyncio.CancelledError | None = None
    try:
        await asyncio.wait(
            {execution_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError as cancellation:
        caller_cancellation = cancellation
        execution_task.cancel("product operation caller cancelled")

    heartbeat_failed = False
    if heartbeat_task.done():
        try:
            heartbeat_task.result()
        except BaseException:
            heartbeat_failed = True
    if heartbeat_failed and not execution_task.done():
        execution_task.cancel("product execution heartbeat failed")

    execution_outcome = await await_shielded_task_outcome(
        execution_task,
        cancellation=caller_cancellation,
    )
    stop_heartbeat.set()
    heartbeat_outcome = await await_shielded_task_outcome(
        heartbeat_task,
        cancellation=execution_outcome.cancellation,
    )

    failures: list[BaseException] = []
    if execution_outcome.error is not None:
        execution_error = execution_outcome.error
        if isinstance(execution_error, asyncio.CancelledError):
            if execution_error.__cause__ is not None:
                failures.append(execution_error.__cause__)
            elif caller_cancellation is None and not heartbeat_failed:
                failures.append(
                    unexpected_child_cancellation_error(
                        execution_error,
                        operation="product operation execution",
                    )
                )
        else:
            failures.append(execution_error)
    if heartbeat_outcome.error is not None and execution_outcome.result is None:
        heartbeat_error = heartbeat_outcome.error
        if isinstance(heartbeat_error, asyncio.CancelledError):
            heartbeat_error = unexpected_child_cancellation_error(
                heartbeat_error,
                operation="product execution heartbeat",
            )
        failures.append(heartbeat_error)

    cancellation = heartbeat_outcome.cancellation
    if cancellation is not None:
        if not failures:
            raise cancellation
        failure: BaseException = (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup(
                "Product operation cancellation retained multiple failures.",
                failures,
            )
        )
        _raise_cancellation_with_failure(
            cancellation,
            failure,
            operation="product operation execution",
        )
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "Product operation execution and heartbeat both failed.",
            failures,
        )
    if execution_outcome.result is None:
        raise RuntimeError("Product operation execution returned no operation.")
    return execution_outcome.result


async def _retain_product_execution_claim(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
) -> None:
    """Renew and fence one claim within a deadline shorter than its lease."""

    async with asyncio.timeout(PRODUCT_EXECUTION_HEARTBEAT_TIMEOUT_SECONDS):
        retained = await _reconcile_ambiguous_product_store_write(
            lambda: service.product_store.heartbeat_execution(
                work_id=operation.work_id,
                claim_id=claim_id,
                lease_seconds=PRODUCT_EXECUTION_LEASE_SECONDS,
            ),
            operation="product execution heartbeat",
        )
    if type(retained) is not bool:
        raise TypeError("Product store returned an invalid execution heartbeat.") from None
    if not retained:
        raise ProductExecutionClaimLost(
            "Product execution ownership was lost before completion."
        ) from None


async def _terminalize_failed_product_operation(
    product_store: ProductOperationStore,
    operation: ProductOperation,
    claim_id: str,
    execution_error: BaseException,
) -> Never:
    """Persist a safe terminal projection without losing execution failure evidence."""

    try:
        await _finish_product_operation_resisting_cancellation(
            product_store,
            work_id=operation.work_id,
            claim_id=claim_id,
            status="failed",
            result=None,
            cancellation=(
                execution_error if isinstance(execution_error, asyncio.CancelledError) else None
            ),
        )
    except asyncio.CancelledError as finalization_cancellation:
        if finalization_cancellation is execution_error:
            raise
        _raise_cancellation_with_failure(
            finalization_cancellation,
            execution_error,
            operation="product operation failure finalization",
        )
    except BaseException as finalization_error:
        if finalization_error is execution_error:
            raise
        raise BaseExceptionGroup(
            "Product operation execution and terminal persistence both failed.",
            [execution_error, finalization_error],
        ) from None
    raise execution_error


async def _finish_product_operation_resisting_cancellation(
    product_store: ProductOperationStore,
    *,
    work_id: str,
    claim_id: str,
    status: Literal["completed", "failed"],
    result: str | None,
    cancellation: asyncio.CancelledError | None = None,
) -> ProductOperation:
    """Settle one terminal product update before propagating caller cancellation."""

    async def finish() -> ProductOperation:
        operation = await _reconcile_ambiguous_product_store_write(
            lambda: product_store.finish(
                work_id=work_id,
                claim_id=claim_id,
                status=status,
                result=result,
            ),
            operation="product operation finalization",
        )
        return _validated_product_store_operation(
            operation,
            source="finish",
            expected={"work_id": work_id, "status": status, "result": result},
            validate_complete=True,
        )

    finalization_task = asyncio.create_task(
        finish(),
        name="cayu-public-operation-finalization",
    )
    outcome = await await_shielded_task_outcome(
        finalization_task,
        cancellation=cancellation,
    )
    if outcome.error is not None:
        finalization_error = outcome.error
        if isinstance(finalization_error, asyncio.CancelledError):
            finalization_error = unexpected_child_cancellation_error(
                finalization_error,
                operation="product operation finalization",
            )
        if outcome.cancellation is not None:
            _raise_cancellation_with_failure(
                outcome.cancellation,
                finalization_error,
                operation="product operation finalization",
            )
        raise finalization_error
    if outcome.result is None:
        raise RuntimeError("Product operation finalization returned no operation.")
    if outcome.cancellation is not None:
        raise outcome.cancellation
    return outcome.result


async def _release_product_execution_claim_resisting_cancellation(
    product_store: ProductOperationStore,
    *,
    work_id: str,
    claim_id: str,
) -> None:
    """Relinquish pending ownership before propagating caller cancellation."""

    async def release() -> bool:
        released = await _reconcile_ambiguous_product_store_write(
            lambda: product_store.release_execution(
                work_id=work_id,
                claim_id=claim_id,
            ),
            operation="product execution claim release",
        )
        if type(released) is not bool:
            raise TypeError("Product store returned an invalid execution-claim release.")
        return released

    release_task = asyncio.create_task(
        release(),
        name="cayu-public-operation-claim-release",
    )
    outcome = await await_shielded_task_outcome(release_task)
    if outcome.error is not None:
        release_error = outcome.error
        if isinstance(release_error, asyncio.CancelledError):
            release_error = unexpected_child_cancellation_error(
                release_error,
                operation="product execution claim release",
            )
        if outcome.cancellation is not None:
            _raise_cancellation_with_failure(
                outcome.cancellation,
                release_error,
                operation="product execution claim release",
            )
        raise release_error
    if outcome.result is None:
        raise RuntimeError("Product execution claim release returned no result.")
    if outcome.cancellation is not None:
        raise outcome.cancellation


async def _reconcile_ambiguous_product_store_write(
    write: Callable[[], Awaitable[Any]],
    *,
    operation: str,
) -> Any:
    """Retry one idempotent store write with the same identity after lost acknowledgement."""

    try:
        return await write()
    except (ProductExecutionClaimLost, ProductOperationSettlementConflict):
        raise
    except asyncio.CancelledError:
        raise
    except Exception as first_failure:
        try:
            return await write()
        except (
            ProductExecutionClaimLost,
            ProductOperationSettlementConflict,
        ) as reconciliation_conflict:
            if reconciliation_conflict.__cause__ is None:
                raise reconciliation_conflict from first_failure
            raise BaseExceptionGroup(
                f"{operation.capitalize()} and acknowledgement reconciliation conflicted.",
                [first_failure, reconciliation_conflict],
            ) from None
        except asyncio.CancelledError as cancellation:
            _raise_cancellation_with_failure(
                cancellation,
                first_failure,
                operation=operation,
            )
        except BaseException as reconciliation_failure:
            if reconciliation_failure.__cause__ is None:
                raise reconciliation_failure from first_failure
            raise BaseExceptionGroup(
                f"{operation.capitalize()} and acknowledgement reconciliation both failed.",
                [first_failure, reconciliation_failure],
            ) from None


def _raise_cancellation_with_failure(
    cancellation: asyncio.CancelledError,
    failure: BaseException,
    *,
    operation: str,
) -> Never:
    cancellation.add_note(f"{operation.capitalize()} also failed: {type(failure).__name__}.")
    prior_cause = cancellation.__cause__
    if prior_cause is not None and prior_cause is not failure:
        failure = BaseExceptionGroup(
            "Product operation cancellation retained multiple failure causes.",
            [prior_cause, failure],
        )
    raise cancellation from failure


def create_agent_service(
    app: CayuApp,
    *,
    agent_name: str,
    mode: ServiceMode | str,
    product_access: ProductAccess,
    operator_access: OperatorAccess,
    product_store: ProductOperationStore,
    product_api_path: str = "/api",
    control_plane_path: str = "/cayu",
) -> CayuService:
    """Assemble the maintained product API and separately protected control plane."""

    if not isinstance(app, CayuApp):
        raise TypeError("create_agent_service requires a CayuApp.")
    agent_name = require_durable_clean_nonblank(agent_name, "agent_name")
    if agent_name not in app.list_agents():
        raise ValueError(f"agent_name must identify a registered agent: {agent_name}")
    mode = ServiceMode(mode)
    if not isinstance(
        product_access,
        (AuthenticatedProductAccess, DevelopmentProductAccess, PlaceholderProductAccess),
    ):
        raise TypeError("product_access must be an explicit Cayu product access policy.")
    if not isinstance(
        operator_access,
        (AuthenticatedAccess, OpenAccess, PlaceholderOperatorAccess),
    ):
        raise TypeError(
            "operator_access must be AuthenticatedAccess, deliberate OpenAccess, "
            "or PlaceholderOperatorAccess."
        )
    if not isinstance(product_store, ProductOperationStore):
        raise TypeError("product_store must implement ProductOperationStore.")
    try:
        store_category = ServiceIdentityStoreKind(product_store.category)
    except (AttributeError, ValueError) as exc:
        raise TypeError("product_store must declare a supported identity-store category.") from exc
    product_api_path = _normalized_service_path(product_api_path, "product_api_path")
    control_plane_path = _normalized_service_path(control_plane_path, "control_plane_path")
    if (
        product_api_path == control_plane_path
        or product_api_path.startswith(f"{control_plane_path}/")
        or control_plane_path.startswith(f"{product_api_path}/")
    ):
        raise ValueError("The product API must be separate from the operator control plane.")

    server = FastAPI(
        title="Cayu public agent service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    product_operations_path = f"{product_api_path}/operations"
    server.add_middleware(
        _ProductBoundaryMiddleware,
        product_operations_path=product_operations_path,
    )
    runtime_session_store = _runtime_store_durability(app.session_store)
    runtime_task_store = (
        "missing" if app.task_store is None else _runtime_store_durability(app.task_store)
    )
    manifest = PublicServiceManifest(
        mode=mode,
        product_access=product_access.kind,
        operator_access=operator_access.kind,
        identity_store=store_category,
        runtime_session_store=runtime_session_store,
        runtime_task_store=runtime_task_store,
        product_api_path=product_api_path,
        control_plane_path=control_plane_path,
    )
    service = CayuService(
        cayu_app=app,
        asgi_app=server,
        manifest=manifest,
        product_store=product_store,
        agent_name=agent_name,
        _assembly_token=_SERVICE_ASSEMBLY_TOKEN,
    )
    auth_dependency = _product_auth_dependency(
        _placeholder_product_auth
        if isinstance(product_access, PlaceholderProductAccess)
        else cast("ProductAuthDependency", product_access.dependency)
    )
    product_auth = Depends(auth_dependency)

    @server.exception_handler(RequestValidationError)
    async def _bounded_validation_error(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return _private_product_error_response(400, _INVALID_PRODUCT_REQUEST_DETAIL)

    @server.exception_handler(Exception)
    async def _private_internal_error(
        _request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        return _private_product_error_response(500, "Internal server error.")

    @server.get("/health", include_in_schema=False)
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @server.post(
        product_operations_path,
        response_model=PublicOperationResponse,
        status_code=201,
    )
    async def create_operation(
        body: CreateOperationRequest,
        request: Request,
        response: Response,
        principal: ProductPrincipal = product_auth,
    ) -> PublicOperationResponse:
        try:
            idempotency_key = require_durable_clean_nonblank(
                request.headers.get("idempotency-key", ""),
                "Idempotency-Key",
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="A valid Idempotency-Key is required.",
            ) from None
        if len(idempotency_key) > 200:
            raise HTTPException(status_code=400, detail="A valid Idempotency-Key is required.")
        durable_product_input = {
            "tenant_id": principal.tenant_id,
            "idempotency_key": idempotency_key,
            "request": body.request,
        }
        if app.redact_json(durable_product_input) != durable_product_input:
            raise HTTPException(
                status_code=400,
                detail=_INVALID_PRODUCT_REQUEST_DETAIL,
            ) from None
        fingerprint = _product_request_fingerprint(
            agent_name=agent_name,
            request_text=body.request,
        )
        candidate_authority = {
            "public_id": f"op_{uuid4().hex}",
            "work_id": f"work_{uuid4().hex}",
            "session_id": f"session_{uuid4().hex}",
            "task_id": f"task_{uuid4().hex}",
        }
        try:
            reservation = await product_store.reserve(
                tenant_id=principal.tenant_id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                public_id=candidate_authority["public_id"],
                work_id=candidate_authority["work_id"],
                session_id=candidate_authority["session_id"],
                task_id=candidate_authority["task_id"],
                request_text=body.request,
            )
        except ProductIdempotencyConflict:
            raise HTTPException(
                status_code=409,
                detail="Idempotency key conflicts with another operation.",
            ) from None
        if not isinstance(reservation, ProductOperationReservation):
            raise TypeError("Product store must return ProductOperationReservation from reserve().")
        if type(reservation.created) is not bool:
            raise TypeError("Product store returned an invalid reservation from reserve().")
        expected_authority: dict[str, object] = {
            "tenant_id": principal.tenant_id,
            "idempotency_key": idempotency_key,
            "request_fingerprint": fingerprint,
        }
        if reservation.created:
            expected_authority.update(candidate_authority)
            expected_authority.update(
                {"request_text": body.request, "status": "pending", "result": None}
            )
        operation = _validated_product_store_operation(
            reservation.operation,
            source="reserve",
            expected=expected_authority,
            validate_complete=reservation.created,
        )
        if operation.status == "pending":
            completed = await service.execute_work(operation.work_id)
            if completed is None:
                raise RuntimeError("Reserved product work disappeared before execution.")
            operation = _validated_product_store_operation(
                completed,
                source="execute_work",
                expected={
                    "tenant_id": operation.tenant_id,
                    "public_id": operation.public_id,
                    "work_id": operation.work_id,
                    "idempotency_key": operation.idempotency_key,
                    "request_fingerprint": operation.request_fingerprint,
                    "session_id": operation.session_id,
                    "task_id": operation.task_id,
                },
                validate_complete=False,
            )
        if not reservation.created:
            response.status_code = 200
        return _public_operation(operation, app=app)

    @server.get(
        f"{product_operations_path}/{{public_id}}",
        response_model=PublicOperationResponse,
    )
    async def read_operation(
        public_id: str,
        principal: ProductPrincipal = product_auth,
    ) -> PublicOperationResponse:
        try:
            public_id = require_durable_clean_nonblank(public_id, "public_id")
        except ValueError:
            raise HTTPException(status_code=404, detail="Operation not found.") from None
        if len(public_id) > MAX_PRODUCT_IDENTITY_CHARS:
            raise HTTPException(status_code=404, detail="Operation not found.")
        operation = await product_store.find(
            tenant_id=principal.tenant_id,
            public_id=public_id,
        )
        if operation is None:
            raise HTTPException(status_code=404, detail="Operation not found.")
        operation = _validated_product_store_operation(
            operation,
            source="find",
            expected={"tenant_id": principal.tenant_id, "public_id": public_id},
            validate_complete=False,
        )
        return _public_operation(operation, app=app)

    from cayu.server import mount_cayu

    operator_mount_access: ServerAccessConfig
    if isinstance(operator_access, PlaceholderOperatorAccess):
        operator_mount_access = AuthenticatedAccess(dependency=_placeholder_operator_auth)
    else:
        operator_mount_access = operator_access
    mount_cayu(
        server,
        app,
        path=control_plane_path,
        access=operator_mount_access,
    )
    server.state.cayu_public_service = service
    server.state.cayu_public_service_manifest = manifest
    _materialize_asgi_middleware_stacks(server, seen=set())
    service._route_signature = _service_route_signature(server)
    return service


_SERVICE_ASSEMBLY_TOKEN = object()


async def _placeholder_product_auth(_request: Request) -> ProductPrincipal:
    raise HTTPException(status_code=503, detail="Product authentication is not configured.")


async def _placeholder_operator_auth(_request: Request) -> Mapping[str, str]:
    raise HTTPException(status_code=503, detail="Operator authentication is not configured.")


def _runtime_store_durability(store: object) -> RuntimeStoreDurability:
    """Normalize explicit store evidence without trusting arbitrary truthy values."""

    declared = inspect.getattr_static(store, "service_durability", None)
    try:
        return RuntimeStoreDurability(declared)
    except (TypeError, ValueError):
        return RuntimeStoreDurability.UNVERIFIED


def _service_route_signature(app: FastAPI) -> tuple[object, ...]:
    """Capture authorization-relevant identities throughout the assembled ASGI graph."""

    return _asgi_signature(app, seen=set())


def _materialize_asgi_middleware_stacks(app: object, *, seen: set[int]) -> None:
    """Build lazy middleware once so normal startup cannot change sealed identity."""

    app_identity = id(app)
    if app_identity in seen:
        return
    seen.add(app_identity)
    routes = getattr(app, "routes", None)
    if routes is None:
        return
    for route in routes:
        mounted_app = getattr(route, "app", None)
        if (
            mounted_app is not None
            and mounted_app is not route
            and getattr(mounted_app, "routes", None) is not None
        ):
            _materialize_asgi_middleware_stacks(mounted_app, seen=seen)
    if getattr(app, "middleware_stack", None) is None:
        build_middleware_stack = getattr(app, "build_middleware_stack", None)
        if callable(build_middleware_stack):
            cast("Any", app).middleware_stack = build_middleware_stack()


def _asgi_signature(app: object, *, seen: set[int]) -> tuple[object, ...]:
    app_identity = id(app)
    if app_identity in seen:
        return ("asgi-cycle", app_identity)
    seen.add(app_identity)
    routes = getattr(app, "routes", None)
    if routes is None:
        return ("opaque-asgi", app_identity, id(type(app)))
    middleware = tuple(
        (
            id(item),
            id(getattr(item, "cls", None)),
            tuple(id(value) for value in getattr(item, "args", ())),
            tuple(
                sorted((str(key), id(value)) for key, value in getattr(item, "kwargs", {}).items())
            ),
        )
        for item in getattr(app, "user_middleware", ())
    )
    exception_handlers = tuple(
        sorted(
            (
                repr(key),
                id(handler),
            )
            for key, handler in getattr(app, "exception_handlers", {}).items()
        )
    )
    dependency_overrides = tuple(
        sorted(
            (id(original), id(replacement))
            for original, replacement in getattr(app, "dependency_overrides", {}).items()
        )
    )
    router = getattr(app, "router", None)
    return (
        "asgi-app",
        app_identity,
        tuple(_route_signature(route, seen=seen) for route in routes),
        middleware,
        exception_handlers,
        dependency_overrides,
        _router_dispatch_signature(router),
        id(getattr(app, "middleware_stack", None)),
    )


def _router_dispatch_signature(router: object) -> tuple[object, ...] | None:
    """Capture mutable router state that can change dispatch during startup."""

    if router is None:
        return None
    return (
        "router-dispatch",
        id(router),
        _callable_identity(getattr(router, "default", None)),
        _callable_identity(getattr(router, "middleware_stack", None)),
        _callable_identity(getattr(router, "lifespan_context", None)),
        tuple(_callable_identity(handler) for handler in getattr(router, "on_startup", ())),
        tuple(_callable_identity(handler) for handler in getattr(router, "on_shutdown", ())),
        getattr(router, "redirect_slashes", None),
    )


def _callable_identity(value: object) -> tuple[object, ...]:
    """Return a stable identity for functions and freshly-bound method objects."""

    bound_function = getattr(value, "__func__", None)
    bound_self = getattr(value, "__self__", None)
    if bound_function is not None and bound_self is not None:
        return ("bound-callable", id(bound_self), id(bound_function))
    return ("callable", id(value))


def _route_signature(route: object, *, seen: set[int]) -> tuple[object, ...]:
    mounted_app = getattr(route, "app", None)
    route_app_signature: tuple[object, ...] | None = None
    if mounted_app is not None and mounted_app is not route:
        if getattr(mounted_app, "routes", None) is not None:
            route_app_signature = _asgi_signature(mounted_app, seen=seen)
        else:
            route_app_signature = ("route-asgi", id(mounted_app), id(type(mounted_app)))
    return (
        "route",
        id(route),
        id(getattr(route, "endpoint", None)),
        getattr(route, "path", None),
        tuple(sorted(getattr(route, "methods", None) or ())),
        getattr(route, "status_code", None),
        id(getattr(route, "response_model", None)),
        id(getattr(route, "response_field", None)),
        id(getattr(route, "body_field", None)),
        id(getattr(route, "response_class", None)),
        id(getattr(route, "dependency_overrides_provider", None)),
        _dependant_signature(getattr(route, "dependant", None), seen=set()),
        route_app_signature,
    )


def _dependant_signature(dependant: object, *, seen: set[int]) -> tuple[object, ...] | None:
    if dependant is None:
        return None
    dependant_identity = id(dependant)
    if dependant_identity in seen:
        return ("dependant-cycle", dependant_identity)
    seen.add(dependant_identity)
    return (
        "dependant",
        dependant_identity,
        id(getattr(dependant, "call", None)),
        getattr(dependant, "use_cache", None),
        tuple(getattr(dependant, "security_scopes", None) or ()),
        tuple(
            _dependant_signature(child, seen=seen)
            for child in getattr(dependant, "dependencies", ())
        ),
    )


def _is_maintained_service(value: object) -> TypeGuard[CayuService]:
    """Validate assembler-owned provenance before check or serve trusts a manifest."""

    if type(value) is not CayuService:
        return False
    service = value
    app = service.asgi_app
    return (
        service._assembly_token is _SERVICE_ASSEMBLY_TOKEN
        and service._route_signature is not None
        and service._route_signature == _service_route_signature(app)
        and getattr(app.state, "cayu_public_service", None) is service
        and getattr(app.state, "cayu_public_service_manifest", None) is service.manifest
        and app.docs_url is None
        and app.redoc_url is None
        and app.openapi_url is None
    )


def _product_auth_dependency(
    dependency: ProductAuthDependency,
) -> Callable[[Request], Awaitable[ProductPrincipal]]:
    async def resolve(request: Request) -> ProductPrincipal:
        if inspect.iscoroutinefunction(dependency):
            result = await dependency(request)
        else:
            result = await run_in_threadpool(dependency, request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ProductPrincipal):
            try:
                return ProductPrincipal(
                    tenant_id=result.tenant_id,
                    subject_id=result.subject_id,
                )
            except (AttributeError, TypeError, ValueError):
                raise TypeError("Product authentication returned an invalid principal.") from None
        if isinstance(result, Mapping):
            return ProductPrincipal.model_validate(dict(result))
        raise TypeError(
            "Product auth dependencies must return ProductPrincipal or a compatible mapping."
        )

    return resolve


def _public_operation(
    operation: ProductOperation,
    *,
    app: CayuApp,
) -> PublicOperationResponse:
    public_id = app.redact_json(operation.public_id)
    if type(public_id) is not str or public_id != operation.public_id:
        raise RuntimeError("Product store returned unsafe public authority.")
    raw_result = operation.result
    if raw_result is not None and (
        type(raw_result) is not str or len(raw_result) > MAX_PUBLIC_RESULT_CHARS
    ):
        raise RuntimeError("Product store returned an invalid public result.")
    result = app.redact_json(raw_result)
    if result is not None and type(result) is not str:
        raise RuntimeError("Product store returned an invalid public result.")
    return PublicOperationResponse(
        id=public_id,
        status=operation.status,
        result=(_bounded_public_result_text(result) if result is not None else None),
    )


def _normalized_service_path(value: str, field_name: str) -> str:
    normalized = normalize_api_path(value, field_name=field_name)
    if normalized != value:
        raise ValueError(
            f"{field_name} must be an absolute normalized non-root path without a trailing slash."
        )
    if "{" in normalized or "}" in normalized:
        raise ValueError(f"{field_name} must be a fixed path without route parameters.")
    return normalized
