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
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic_core import InitErrorDetails
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
)
from cayu.core.events import Event, EventType
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import Message
from cayu.runtime.app import CayuApp
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    TaskExecutionSource,
)
from cayu.runtime.loop_policies import BeforeStopContext, BeforeStopDecision, LoopPolicy
from cayu.runtime.service_manifest import (
    PublicServiceManifest,
    RuntimeStoreDurability,
    ServiceIdentityStoreKind,
    ServiceMode,
)
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskStatus,
    task_create_with_runtime_invocation,
)
from cayu.runtime.usage import is_conversational_model_completion_payload
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
MAX_PRODUCT_RESULT_EVIDENCE_SCAN_EVENTS = 256

_PRODUCT_RESULT_RECEIPT_RECORD_TYPE = "cayu.product-result-receipt"
_PRODUCT_RESULT_RECEIPT_SCHEMA_VERSION = 1
_PRODUCT_PUBLICATION_EXECUTION_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="cayu.server:product-publication",
    behavior_version="1",
    implementation_version="1",
)
_PRODUCT_RECOVERY_MESSAGE = (
    "Continue this interrupted operation from its durable session state. "
    "Do not repeat work whose outcome is already recorded."
)

ProductRecoveryStatus = Literal[
    "runtime_active",
    "waiting_for_approval",
    "waiting_for_user_input",
    "interrupted",
    "manual_reconciliation_required",
]

_ProductExecutionResult = TypeVar("_ProductExecutionResult")

_INVALID_PRODUCT_REQUEST_DETAIL = "Invalid product request."
_OVERSIZED_PRODUCT_REQUEST_DETAIL = "Product request exceeds the server byte limit."


class _ProductWorkReconciliationRequired(RuntimeError):
    """Durable Cayu work may exist and must not be terminalized or redispatched."""


class _ProductRuntimeObservationUncertain(RuntimeError):
    """The product observer cannot prove the durable runtime outcome."""


def _product_request_fingerprint(*, agent_name: str, request_text: str) -> str:
    fingerprint_input = json.dumps(
        {"agent_name": agent_name, "request": request_text, "schema_version": 1},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(fingerprint_input).hexdigest()


def _product_result_receipt_digest(
    *,
    work_id: str,
    public_id: str,
    request_fingerprint: str,
    session_id: str,
    task_id: str,
    source_event_id: str,
    source_event_sequence: int,
    model_step_id: str,
    model_attempt_id: str,
    interaction_id: str,
    publication_status: Literal["completed", "failed"],
    result: str | None,
    failure_code: Literal["unsafe_result"] | None,
) -> str:
    document = {
        "interaction_id": interaction_id,
        "public_id": public_id,
        "record_type": _PRODUCT_RESULT_RECEIPT_RECORD_TYPE,
        "request_fingerprint": request_fingerprint,
        "result": result,
        "schema_version": _PRODUCT_RESULT_RECEIPT_SCHEMA_VERSION,
        "session_id": session_id,
        "source_event_id": source_event_id,
        "source_event_sequence": source_event_sequence,
        "model_step_id": model_step_id,
        "model_attempt_id": model_attempt_id,
        "publication_status": publication_status,
        "failure_code": failure_code,
        "task_id": task_id,
        "work_id": work_id,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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


class ProductResultReceipt(BaseModel):
    """Application-owned proof of one exact public result projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.product-result-receipt"] = _PRODUCT_RESULT_RECEIPT_RECORD_TYPE
    schema_version: Literal[1] = _PRODUCT_RESULT_RECEIPT_SCHEMA_VERSION
    work_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    public_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    request_fingerprint: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    session_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    task_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    source_event_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    source_event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    model_step_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    model_attempt_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    interaction_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    publication_status: Literal["completed", "failed"]
    result: str | None = Field(max_length=MAX_PUBLIC_RESULT_CHARS)
    failure_code: Literal["unsafe_result"] | None = None
    receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        work_id: str,
        public_id: str,
        request_fingerprint: str,
        session_id: str,
        task_id: str,
        source_event_id: str,
        source_event_sequence: int,
        model_step_id: str,
        model_attempt_id: str,
        interaction_id: str,
        publication_status: Literal["completed", "failed"],
        result: str | None,
        failure_code: Literal["unsafe_result"] | None = None,
    ) -> ProductResultReceipt:
        return cls(
            work_id=work_id,
            public_id=public_id,
            request_fingerprint=request_fingerprint,
            session_id=session_id,
            task_id=task_id,
            source_event_id=source_event_id,
            source_event_sequence=source_event_sequence,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            interaction_id=interaction_id,
            publication_status=publication_status,
            result=result,
            failure_code=failure_code,
            receipt_digest=_product_result_receipt_digest(
                work_id=work_id,
                public_id=public_id,
                request_fingerprint=request_fingerprint,
                session_id=session_id,
                task_id=task_id,
                source_event_id=source_event_id,
                source_event_sequence=source_event_sequence,
                model_step_id=model_step_id,
                model_attempt_id=model_attempt_id,
                interaction_id=interaction_id,
                publication_status=publication_status,
                result=result,
                failure_code=failure_code,
            ),
        )

    @field_validator(
        "work_id",
        "public_id",
        "request_fingerprint",
        "session_id",
        "task_id",
        "source_event_id",
        "model_step_id",
        "model_attempt_id",
        "interaction_id",
    )
    @classmethod
    def validate_clean_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("result")
    @classmethod
    def validate_result(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_text(value, "result")

    @model_validator(mode="after")
    def validate_digest(self) -> ProductResultReceipt:
        if self.publication_status == "completed":
            if self.result is None or self.failure_code is not None:
                raise ValueError("Completed publication receipts require only a result.")
        elif self.result is not None or self.failure_code != "unsafe_result":
            raise ValueError("Failed publication receipts require an unsafe-result code.")
        expected = _product_result_receipt_digest(
            work_id=self.work_id,
            public_id=self.public_id,
            request_fingerprint=self.request_fingerprint,
            session_id=self.session_id,
            task_id=self.task_id,
            source_event_id=self.source_event_id,
            source_event_sequence=self.source_event_sequence,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
            interaction_id=self.interaction_id,
            publication_status=self.publication_status,
            result=self.result,
            failure_code=self.failure_code,
        )
        if self.receipt_digest != expected:
            raise ValueError("receipt_digest does not match the result receipt content.")
        return self


class ProductOperation(BaseModel):
    """Application-owned binding between public identity and private Cayu authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tenant_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    subject_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    public_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    work_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    idempotency_key: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    request_fingerprint: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    session_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    task_id: str = Field(max_length=MAX_PRODUCT_IDENTITY_CHARS)
    request_text: str = Field(max_length=100_000)
    status: Literal["pending", "completed", "failed"]
    result: str | None = Field(max_length=100_000)
    result_receipt: ProductResultReceipt | None = None
    recovery_status: ProductRecoveryStatus | None = None

    @field_validator(
        "tenant_id",
        "subject_id",
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

    @model_validator(mode="after")
    def validate_result_state(self) -> ProductOperation:
        receipt = self.result_receipt
        if receipt is not None and (
            receipt.work_id != self.work_id
            or receipt.public_id != self.public_id
            or receipt.request_fingerprint != self.request_fingerprint
            or receipt.session_id != self.session_id
            or receipt.task_id != self.task_id
        ):
            raise ValueError("result_receipt does not belong to this product operation.")
        if self.status == "pending" and self.result is not None:
            raise ValueError("Pending product operations cannot expose a terminal result.")
        if self.status == "completed" and (
            receipt is None
            or receipt.publication_status != "completed"
            or self.result != receipt.result
        ):
            raise ValueError("Completed product operations require their exact result receipt.")
        if self.status == "failed" and self.result is not None:
            raise ValueError("Failed product operations cannot expose a result.")
        if self.status != "pending" and self.recovery_status is not None:
            raise ValueError("Terminal product operations cannot retain recovery status.")
        return self


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
) -> ProductOperation:
    """Snapshot and fully validate one store-supplied product operation."""

    if not isinstance(value, ProductOperation):
        raise TypeError(f"Product store must return ProductOperation from {source}().")
    try:
        fields = {
            field_name: getattr(value, field_name) for field_name in ProductOperation.model_fields
        }
        operation = ProductOperation.model_validate(fields)
    except (AttributeError, TypeError, ValueError):
        raise TypeError(f"Product store returned an invalid operation from {source}().") from None
    if any(
        type(getattr(operation, field_name)) is not type(expected_value)
        or getattr(operation, field_name) != expected_value
        for field_name, expected_value in expected.items()
    ):
        raise RuntimeError(f"Product store returned inconsistent authority from {source}().")
    return operation


def _validated_product_result_receipt(
    value: object,
    *,
    expected: ProductResultReceipt,
    source: str,
) -> ProductResultReceipt:
    if not isinstance(value, ProductResultReceipt):
        raise TypeError(f"Product store must return ProductResultReceipt from {source}().")
    try:
        receipt = ProductResultReceipt.model_validate(
            {
                field_name: getattr(value, field_name)
                for field_name in ProductResultReceipt.model_fields
            }
        )
    except (AttributeError, TypeError, ValueError):
        raise TypeError(f"Product store returned an invalid receipt from {source}().") from None
    if receipt != expected:
        raise RuntimeError(f"Product store returned inconsistent receipt from {source}().")
    return receipt


class ProductIdempotencyConflict(Exception):
    """An idempotency identity is already bound to different trusted work."""


class ProductExecutionClaimLost(Exception):
    """A product worker no longer owns the operation it attempted to settle."""


class ProductOperationSettlementConflict(Exception):
    """A product operation already has a different terminal result."""


class ProductResultReceiptConflict(Exception):
    """Product work is already bound to a different result receipt."""


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
        subject_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation: ...

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None: ...

    async def find_by_session_id(self, *, session_id: str) -> ProductOperation | None:
        """Load private product authority for an internal continuation hook.

        This lookup is never exposed as a product API and must use the exact
        application-owned unique session index rather than Cayu labels,
        metadata, or a scan.
        """
        ...

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

    async def record_result_receipt(
        self,
        *,
        work_id: str,
        claim_id: str,
        receipt: ProductResultReceipt,
    ) -> ProductResultReceipt:
        """Insert or reconstruct the exact result receipt owned by ``claim_id``.

        Repeating the same content-bound receipt after an ambiguous
        acknowledgement must return the existing receipt. While work remains
        pending, its current execution owner may replace a candidate only with
        one carrying a greater durable source-event sequence. This permits a
        queued message to advance the same session after an earlier before-stop
        candidate without allowing stale workers to overwrite newer evidence.
        """
        ...

    async def record_recovery_status(
        self,
        *,
        work_id: str,
        claim_id: str,
        recovery_status: ProductRecoveryStatus,
    ) -> ProductOperation:
        """Record bounded pending-work evidence under the current execution claim.

        Repeating the same status after an ambiguous acknowledgement must
        reconstruct the current operation. Terminal work and stale claims must
        reject the update.
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
        """Conditionally settle owned work or reconstruct the same terminal write.

        Completed settlement must match the operation's recorded result receipt.
        Failed settlement must persist no public result.
        """
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
    recovery_status: ProductRecoveryStatus | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


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

    async def _continuation_loop_policies(
        self,
        session_id: str,
    ) -> tuple[LoopPolicy, ...]:
        operation = await self.product_store.find_by_session_id(session_id=session_id)
        if operation is None:
            return ()
        operation = _validated_product_store_operation(
            operation,
            source="find_by_session_id",
            expected={"session_id": session_id},
        )
        if operation.request_fingerprint != _product_request_fingerprint(
            agent_name=self.agent_name,
            request_text=operation.request_text,
        ):
            raise RuntimeError(
                "Product continuation does not match this service's agent and request contract."
            )
        return (_ProductContinuationReceiptPolicy(service=self, operation=operation),)

    async def execute_work(self, work_id: str) -> ProductOperation | None:
        """Reload trusted ownership from product storage, then run opaque queued work."""

        return await self._execute_work(work_id, expected_operation=None)

    async def _execute_work(
        self,
        work_id: str,
        *,
        expected_operation: ProductOperation | None,
    ) -> ProductOperation | None:
        """Execute work, optionally retaining authority already accepted by this request."""

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
        expected_claim_authority: dict[str, object] = {"work_id": work_id}
        if expected_operation is not None:
            expected_claim_authority = _product_operation_authority_fields(expected_operation)
        operation = _validated_product_store_operation(
            claim.operation,
            source="claim_execution",
            expected=expected_claim_authority,
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
                await _release_product_claim_after_failure(
                    product_store=self.product_store,
                    operation=operation,
                    claim_id=claim_id,
                    failure=reconciliation_error,
                    action="product operation reconciliation handoff",
                )
            except (BaseExceptionGroup, Exception, asyncio.CancelledError) as execution_error:
                await _terminalize_failed_product_operation(
                    self.product_store,
                    operation,
                    claim_id,
                    execution_error,
                )
            if not task_is_ready:
                try:
                    reconciled = await _recover_progressed_product_work(
                        service=self,
                        operation=operation,
                        claim_id=claim_id,
                    )
                except BaseException as recovery_error:
                    await _release_product_claim_after_failure(
                        product_store=self.product_store,
                        operation=operation,
                        claim_id=claim_id,
                        failure=recovery_error,
                        action="product operation recovery handoff",
                    )
                if isinstance(reconciled, ProductOperation):
                    return reconciled
                await _record_product_recovery_status(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                    recovery_status=reconciled,
                )
                return await _release_and_reload_pending_product_operation(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                )
            try:
                await _retain_product_execution_claim(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                )
                receipt_policy = _ProductResultReceiptPolicy(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                )
                (
                    outcome_status,
                    result_receipt,
                    recovery_status,
                ) = await _run_product_operation_bounded(
                    self.cayu_app,
                    RunRequest(
                        agent_name=self.agent_name,
                        messages=[Message.text("user", operation.request_text)],
                        session_id=operation.session_id,
                        task_id=operation.task_id,
                        loop_policies=(receipt_policy,),
                    ),
                    receipt_policy=receipt_policy,
                )
            except _ProductRuntimeObservationUncertain as observation_failure:
                return await _reconcile_uncertain_product_stream(
                    service=self,
                    operation=operation,
                    claim_id=claim_id,
                    observation_failure=observation_failure,
                )
            except (BaseExceptionGroup, Exception, asyncio.CancelledError) as execution_error:
                await _terminalize_failed_product_operation(
                    self.product_store,
                    operation,
                    claim_id,
                    execution_error,
                )
            return await _settle_product_run_outcome(
                service=self,
                operation=operation,
                claim_id=claim_id,
                outcome_status=outcome_status,
                result_receipt=result_receipt,
                recovery_status=recovery_status,
            )

        return await _run_product_operation_with_heartbeat(
            service=self,
            operation=operation,
            claim_id=claim_id,
            execute_and_settle=execute_and_settle,
        )


async def _settle_product_run_outcome(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    outcome_status: SessionStatus,
    result_receipt: ProductResultReceipt | None,
    recovery_status: ProductRecoveryStatus | None,
) -> ProductOperation:
    if outcome_status is SessionStatus.INTERRUPTED:
        await _record_product_recovery_status(
            service=service,
            operation=operation,
            claim_id=claim_id,
            recovery_status=recovery_status or "interrupted",
        )
        return await _release_and_reload_pending_product_operation(
            service=service,
            operation=operation,
            claim_id=claim_id,
        )
    if outcome_status is SessionStatus.COMPLETED:
        if result_receipt is None:
            raise RuntimeError("Completed product work has no publication receipt.")
        status = result_receipt.publication_status
        result = result_receipt.result if status == "completed" else None
    elif outcome_status is SessionStatus.FAILED:
        status = "failed"
        result = None
    else:
        raise RuntimeError(f"Unsupported product runtime outcome: {outcome_status.value}.")
    return await _finish_product_operation_resisting_cancellation(
        service.product_store,
        operation=operation,
        claim_id=claim_id,
        status=status,
        result=result,
    )


async def _reconcile_uncertain_product_stream(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    observation_failure: _ProductRuntimeObservationUncertain,
) -> ProductOperation:
    """Settle only exact terminal evidence after the runtime stream becomes uncertain."""

    try:
        reconciled = await _reconcile_terminal_product_work(
            service=service,
            operation=operation,
            claim_id=claim_id,
        )
        if reconciled is not None:
            return reconciled
        await _record_product_recovery_status(
            service=service,
            operation=operation,
            claim_id=claim_id,
            recovery_status="manual_reconciliation_required",
        )
        return await _release_and_reload_pending_product_operation(
            service=service,
            operation=operation,
            claim_id=claim_id,
        )
    except BaseException as reconciliation_failure:
        if isinstance(reconciliation_failure, asyncio.CancelledError):
            reconciliation_failure.add_note(
                "Product runtime observation also failed before cancellation."
            )
            prior_cause = reconciliation_failure.__cause__
            reconciliation_failure.__cause__ = (
                observation_failure
                if prior_cause is None
                else BaseExceptionGroup(
                    "Product stream reconciliation retained multiple failure causes.",
                    [prior_cause, observation_failure],
                )
            )
            combined_failure: BaseException = reconciliation_failure
        else:
            combined_failure = BaseExceptionGroup(
                "Product runtime observation and durable reconciliation both failed.",
                [observation_failure, reconciliation_failure],
            )
        await _release_product_claim_after_failure(
            product_store=service.product_store,
            operation=operation,
            claim_id=claim_id,
            failure=combined_failure,
            action="product runtime observation reconciliation",
        )


async def _release_and_reload_pending_product_operation(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
) -> ProductOperation:
    await _release_product_execution_claim_resisting_cancellation(
        service.product_store,
        work_id=operation.work_id,
        claim_id=claim_id,
    )
    reloaded = await service.product_store.find(
        tenant_id=operation.tenant_id,
        public_id=operation.public_id,
    )
    if reloaded is None:
        raise RuntimeError("Product work disappeared after execution-claim release.")
    return _validated_product_store_operation(
        reloaded,
        source="find",
        expected={
            **_product_operation_authority_fields(operation),
        },
    )


async def _record_product_recovery_status(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    recovery_status: ProductRecoveryStatus,
) -> ProductOperation:
    stored = await _reconcile_ambiguous_product_store_write(
        lambda: service.product_store.record_recovery_status(
            work_id=operation.work_id,
            claim_id=claim_id,
            recovery_status=recovery_status,
        ),
        operation="product recovery status",
    )
    return _validated_product_store_operation(
        stored,
        source="record_recovery_status",
        expected={
            **_product_operation_authority_fields(operation),
            "status": "pending",
            "recovery_status": recovery_status,
        },
    )


async def _run_product_operation_bounded(
    app: CayuApp,
    request: RunRequest | ResumeRequest,
    *,
    receipt_policy: _ProductResultReceiptPolicy,
) -> tuple[
    SessionStatus,
    ProductResultReceipt | None,
    ProductRecoveryStatus | None,
]:
    """Consume one run without retaining its complete event or delta history."""

    status: SessionStatus | None = None
    recovery_status: ProductRecoveryStatus | None = None
    if type(request) is RunRequest:
        stream = cast("AsyncGenerator[Event, None]", app.run(request))
    elif type(request) is ResumeRequest:
        stream = cast("AsyncGenerator[Event, None]", app.resume(request))
    else:
        raise TypeError("Product execution requires an exact run or resume request.")
    try:
        async with aclosing(stream):
            async for event in stream:
                receipt_policy.observe(event)
                if event.type == EventType.SESSION_COMPLETED:
                    status = SessionStatus.COMPLETED
                elif event.type == EventType.SESSION_FAILED:
                    status = SessionStatus.FAILED
                elif event.type == EventType.SESSION_INTERRUPTED:
                    status = SessionStatus.INTERRUPTED
                    interruption_type = (event.payload or {}).get("interruption_type")
                    if interruption_type == "tool_approval_required":
                        recovery_status = "waiting_for_approval"
                    elif interruption_type == "user_input_required":
                        recovery_status = "waiting_for_user_input"
                    else:
                        recovery_status = "interrupted"
    except Exception as observation_failure:
        raise _ProductRuntimeObservationUncertain(
            "Product runtime stream ended without an authoritative observed outcome."
        ) from observation_failure
    finally:
        receipt_policy.close()
    if status is None:
        raise _ProductRuntimeObservationUncertain(
            "Product runtime stream ended without an authoritative terminal event."
        )
    if status is SessionStatus.COMPLETED and receipt_policy.receipt is None:
        raise _ProductRuntimeObservationUncertain(
            "Completed product runtime work has no durable publication receipt."
        )
    return status, receipt_policy.receipt, recovery_status


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


def _product_operation_authority_fields(operation: ProductOperation) -> dict[str, object]:
    """Return every immutable product-store authority field."""

    return {
        "tenant_id": operation.tenant_id,
        "subject_id": operation.subject_id,
        "public_id": operation.public_id,
        "work_id": operation.work_id,
        "idempotency_key": operation.idempotency_key,
        "request_fingerprint": operation.request_fingerprint,
        "session_id": operation.session_id,
        "task_id": operation.task_id,
        "request_text": operation.request_text,
    }


def _product_operation_loop_policy_authority(operation: ProductOperation) -> dict[str, object]:
    """Return immutable product authority used by continuation policy behavior."""

    return _product_operation_authority_fields(operation)


class _ProductResultReceiptPolicy(LoopPolicy):
    """Publish the final public result before Cayu commits session completion."""

    def __init__(
        self,
        *,
        service: CayuService,
        operation: ProductOperation,
        claim_id: str,
    ) -> None:
        self._service = service
        self._operation = operation
        self._claim_id = claim_id
        self._capture = _BoundedProductResultCapture(service.cayu_app)
        self._captured_result: str | None = None
        self._receipt: ProductResultReceipt | None = None

    @property
    def name(self) -> str:
        return "product-result-publication"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _PRODUCT_PUBLICATION_EXECUTION_PROFILE_IDENTITY

    @property
    def adoption_replay_identity(self) -> str:
        material = json.dumps(
            {
                "claim_id": self._claim_id,
                "operation_authority": _product_operation_loop_policy_authority(self._operation),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"product-result-publication:v1:sha256:{hashlib.sha256(material).hexdigest()}"

    @property
    def receipt(self) -> ProductResultReceipt | None:
        return self._receipt

    def observe(self, event: Event) -> None:
        payload = event.payload or {}
        if event.type == EventType.MODEL_STARTED:
            self._capture.abort()
            self._capture = _BoundedProductResultCapture(self._service.cayu_app)
            self._captured_result = None
        elif event.type == EventType.MODEL_TEXT_DELTA:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._capture.append(delta)
        elif event.type == EventType.MODEL_COMPLETED:
            self._captured_result = self._capture.finish_complete()

    def close(self) -> None:
        self._capture.abort()

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        result = self._captured_result
        if self._capture.failed:
            publication_status: Literal["completed", "failed"] = "failed"
            failure_code: Literal["unsafe_result"] | None = "unsafe_result"
            result = None
        else:
            redacted_result = self._service.cayu_app.redact_json(context.step_result.text_content)
            if (
                result is None
                or type(redacted_result) is not str
                or len(result) > MAX_PUBLIC_RESULT_CHARS
                or self._service.cayu_app.redact_json(result) != result
                or _bounded_public_result_text(redacted_result) != result
            ):
                raise RuntimeError("Product result does not match its safe runtime projection.")
            publication_status = "completed"
            failure_code = None
        if result is not None and self._service.cayu_app.redact_json(result) != result:
            raise RuntimeError("Product result does not match its safe runtime projection.")
        self._receipt = await _record_product_result_receipt(
            service=self._service,
            operation=self._operation,
            claim_id=self._claim_id,
            model_step_id=context.step_result.model_step_id,
            model_attempt_id=context.step_result.model_attempt_id,
            publication_status=publication_status,
            result=result,
            failure_code=failure_code,
        )
        return BeforeStopDecision.complete(reason="product result receipt recorded")


class _ProductContinuationReceiptPolicy(LoopPolicy):
    """Publish a receipt for operator-driven continuation of product work."""

    def __init__(self, *, service: CayuService, operation: ProductOperation) -> None:
        self._service = service
        self._operation = operation

    @property
    def name(self) -> str:
        return "product-continuation-publication"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _PRODUCT_PUBLICATION_EXECUTION_PROFILE_IDENTITY

    @property
    def adoption_replay_identity(self) -> str:
        material = json.dumps(
            {
                "operation_authority": _product_operation_loop_policy_authority(self._operation),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"product-continuation-publication:v1:sha256:{hashlib.sha256(material).hexdigest()}"

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        if context.session.id != self._operation.session_id:
            raise RuntimeError("Product continuation policy received a different session.")
        claim_id = f"claim_{uuid4().hex}"
        claim = await _reconcile_ambiguous_product_store_write(
            lambda: self._service.product_store.claim_execution(
                work_id=self._operation.work_id,
                claim_id=claim_id,
                lease_seconds=PRODUCT_EXECUTION_LEASE_SECONDS,
            ),
            operation="product continuation execution claim",
        )
        if not isinstance(claim, ProductOperationExecutionClaim):
            raise TypeError(
                "Product store must return ProductOperationExecutionClaim from claim_execution()."
            )
        if type(claim.acquired) is not bool:
            raise TypeError("Product store returned an invalid execution claim.")
        claimed_operation = _validated_product_store_operation(
            claim.operation,
            source="claim_execution",
            expected={
                **_product_operation_authority_fields(self._operation),
            },
        )
        if claimed_operation.status != "pending":
            return BeforeStopDecision.complete(reason="product operation already terminal")
        if not claim.acquired:
            return BeforeStopDecision.interrupt(
                "product publication claim is held by another worker"
            )

        capture = _BoundedProductResultCapture(self._service.cayu_app)
        try:
            capture.append(context.step_result.text_content)
            result = capture.finish_complete()
            if capture.failed:
                publication_status: Literal["completed", "failed"] = "failed"
                failure_code: Literal["unsafe_result"] | None = "unsafe_result"
                result = None
            else:
                if result is None or self._service.cayu_app.redact_json(result) != result:
                    raise RuntimeError(
                        "Product continuation result does not match its safe projection."
                    )
                publication_status = "completed"
                failure_code = None

            async def publish_receipt() -> ProductResultReceipt:
                await _retain_product_execution_claim(
                    service=self._service,
                    operation=claimed_operation,
                    claim_id=claim_id,
                )
                return await _record_product_result_receipt(
                    service=self._service,
                    operation=claimed_operation,
                    claim_id=claim_id,
                    model_step_id=context.step_result.model_step_id,
                    model_attempt_id=context.step_result.model_attempt_id,
                    publication_status=publication_status,
                    result=result,
                    failure_code=failure_code,
                )

            await _run_product_operation_with_heartbeat(
                service=self._service,
                operation=claimed_operation,
                claim_id=claim_id,
                execute_and_settle=publish_receipt,
            )
        except BaseException as publication_error:
            await _release_product_claim_after_failure(
                product_store=self._service.product_store,
                operation=claimed_operation,
                claim_id=claim_id,
                failure=publication_error,
                action="product continuation publication cleanup",
            )
        finally:
            capture.abort()
        await _release_product_execution_claim_resisting_cancellation(
            self._service.product_store,
            work_id=claimed_operation.work_id,
            claim_id=claim_id,
        )
        return BeforeStopDecision.complete(reason="product continuation receipt recorded")


async def _record_product_result_receipt(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    model_step_id: str,
    model_attempt_id: str,
    publication_status: Literal["completed", "failed"],
    result: str | None,
    failure_code: Literal["unsafe_result"] | None,
) -> ProductResultReceipt:
    """Bind one safe public result to its exact durable conversational completion."""

    records = await service.cayu_app.session_store.query_events(
        EventQuery(
            session_id=operation.session_id,
            event_type=EventType.MODEL_COMPLETED,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if len(records) != 1:
        raise RuntimeError("Product result source event is not durably available.")
    source = records[0]
    durable_event = source.event
    interaction_id = durable_event.interaction_id
    if (
        durable_event.session_id != operation.session_id
        or durable_event.agent_name != service.agent_name
        or not is_conversational_model_completion_payload(durable_event.payload)
        or durable_event.payload.get("model_step_id") != model_step_id
        or durable_event.payload.get("model_attempt_id") != model_attempt_id
        or type(interaction_id) is not str
    ):
        raise RuntimeError("Product result source event conflicts with durable runtime evidence.")
    interaction_id = require_durable_clean_nonblank(interaction_id, "interaction_id")
    receipt = ProductResultReceipt.create(
        work_id=operation.work_id,
        public_id=operation.public_id,
        request_fingerprint=operation.request_fingerprint,
        session_id=operation.session_id,
        task_id=operation.task_id,
        source_event_id=durable_event.id,
        source_event_sequence=source.sequence,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        interaction_id=interaction_id,
        publication_status=publication_status,
        result=result,
        failure_code=failure_code,
    )
    stored = await _reconcile_ambiguous_product_store_write(
        lambda: service.product_store.record_result_receipt(
            work_id=operation.work_id,
            claim_id=claim_id,
            receipt=receipt,
        ),
        operation="product result receipt",
    )
    return _validated_product_result_receipt(
        stored,
        expected=receipt,
        source="record_result_receipt",
    )


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
    request = task_create_with_runtime_invocation(
        request,
        source=TaskExecutionSource.PRODUCT_OPERATION,
        verified_origin=InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject=operation.subject_id,
            tenant=operation.tenant_id,
        ),
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

    task = _validated_product_task(
        task,
        operation=operation,
        agent_name=agent_name,
    )
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


def _validated_product_task(
    task: object,
    *,
    operation: ProductOperation,
    agent_name: str,
) -> Task:
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
        or task.invocation.origin
        != InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject=operation.subject_id,
            tenant=operation.tenant_id,
        )
        or task.invocation.root_session_id != operation.session_id
        or task.invocation.source is not TaskExecutionSource.PRODUCT_OPERATION
    ):
        raise RuntimeError("Existing Cayu task conflicts with product work authority.")
    return task


async def _recover_progressed_product_work(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
) -> ProductOperation | ProductRecoveryStatus:
    """Reconcile terminal work or continue one safely abandoned session."""

    await _retain_product_execution_claim(
        service=service,
        operation=operation,
        claim_id=claim_id,
    )
    reconciled = await _reconcile_terminal_product_work(
        service=service,
        operation=operation,
        claim_id=claim_id,
    )
    if reconciled is not None:
        return reconciled
    if await service.cayu_app.session_store.load_state(operation.session_id) is None:
        return "manual_reconciliation_required"

    recovery = await service.cayu_app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=operation.session_id,
            reason="product_worker_recovered_incomplete_session",
        )
    )
    reconciled = await _reconcile_terminal_product_work(
        service=service,
        operation=operation,
        claim_id=claim_id,
    )
    if reconciled is not None:
        return reconciled
    if recovery.pending_approval_id is not None:
        return "waiting_for_approval"
    if recovery.pending_user_input_id is not None:
        return "waiting_for_user_input"
    if IncompleteSessionRecoveryAction.SKIPPED_ACTIVE in recovery.actions:
        return "runtime_active"
    if (
        recovery.status is not SessionStatus.INTERRUPTED
        or IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED not in recovery.actions
    ):
        if recovery.status is SessionStatus.INTERRUPTED:
            return "interrupted"
        return "manual_reconciliation_required"

    await _retain_product_execution_claim(
        service=service,
        operation=operation,
        claim_id=claim_id,
    )
    receipt_policy = _ProductResultReceiptPolicy(
        service=service,
        operation=operation,
        claim_id=claim_id,
    )
    try:
        outcome_status, result_receipt, recovery_status = await _run_product_operation_bounded(
            service.cayu_app,
            ResumeRequest(
                session_id=operation.session_id,
                messages=[Message.text("user", _PRODUCT_RECOVERY_MESSAGE)],
                loop_policies=(receipt_policy,),
            ),
            receipt_policy=receipt_policy,
        )
    except _ProductRuntimeObservationUncertain as observation_failure:
        return await _reconcile_uncertain_product_stream(
            service=service,
            operation=operation,
            claim_id=claim_id,
            observation_failure=observation_failure,
        )
    return await _settle_product_run_outcome(
        service=service,
        operation=operation,
        claim_id=claim_id,
        outcome_status=outcome_status,
        result_receipt=result_receipt,
        recovery_status=recovery_status,
    )


async def _reconcile_terminal_product_work(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
) -> ProductOperation | None:
    """Settle terminal Cayu work from bounded evidence without redispatch."""

    task_store = service.cayu_app.task_store
    if task_store is None:
        raise RuntimeError("task_store is required to reconcile product work.")
    task = await task_store.load_task(operation.task_id)
    if task is None:
        return None
    task = _validated_product_task(
        task,
        operation=operation,
        agent_name=service.agent_name,
    )
    state = await service.cayu_app.session_store.load_state(operation.session_id)
    if state is None:
        return None
    if not getattr(
        service.cayu_app.session_store,
        "supports_terminal_session_evidence",
        False,
    ):
        return None
    if state.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        return None
    expected_task_status = (
        TaskStatus.COMPLETED if state.status is SessionStatus.COMPLETED else TaskStatus.FAILED
    )
    if task.status is not expected_task_status:
        return None
    try:
        evidence = await service.cayu_app.session_store.load_terminal_session_evidence(
            operation.session_id
        )
    except (NotImplementedError, TerminalSessionEvidenceError):
        return None
    if (
        evidence.session.id != operation.session_id
        or evidence.session.agent_name != service.agent_name
        or evidence.session.status is not state.status
        or evidence.terminal_event.event.session_id != operation.session_id
    ):
        return None
    if state.status is SessionStatus.FAILED:
        if evidence.terminal_event.event.type != EventType.SESSION_FAILED:
            return None
        return await _finish_product_operation_resisting_cancellation(
            service.product_store,
            operation=operation,
            claim_id=claim_id,
            status="failed",
            result=None,
        )
    receipt = operation.result_receipt
    if receipt is None or not await _terminal_evidence_matches_product_receipt(
        service=service,
        operation=operation,
        receipt=receipt,
        evidence=evidence,
    ):
        return None
    return await _finish_product_operation_resisting_cancellation(
        service.product_store,
        operation=operation,
        claim_id=claim_id,
        status=receipt.publication_status,
        result=(receipt.result if receipt.publication_status == "completed" else None),
    )


async def _terminal_evidence_matches_product_receipt(
    *,
    service: CayuService,
    operation: ProductOperation,
    receipt: ProductResultReceipt,
    evidence: TerminalSessionEvidence,
) -> bool:
    if (
        evidence.terminal_event.event.type != EventType.SESSION_COMPLETED
        or receipt.work_id != operation.work_id
        or receipt.public_id != operation.public_id
        or receipt.request_fingerprint != operation.request_fingerprint
        or receipt.session_id != operation.session_id
        or receipt.task_id != operation.task_id
    ):
        return False
    records = await service.cayu_app.session_store.query_events(
        EventQuery(
            session_id=operation.session_id,
            event_id=receipt.source_event_id,
            order_by=EventOrder.SEQUENCE_ASC,
            limit=2,
        )
    )
    if len(records) != 1:
        return False
    source = records[0]
    event = source.event
    if (
        event.type != EventType.MODEL_COMPLETED
        or event.session_id != operation.session_id
        or event.agent_name != service.agent_name
        or not is_conversational_model_completion_payload(event.payload)
        or event.payload.get("model_step_id") != receipt.model_step_id
        or event.payload.get("model_attempt_id") != receipt.model_attempt_id
        or source.sequence != receipt.source_event_sequence
        or event.interaction_id != receipt.interaction_id
        or source.sequence >= evidence.boundary.terminal_event_sequence
    ):
        return False
    try:
        latest_completion = await _latest_conversational_model_completion(
            service=service,
            session_id=operation.session_id,
            before_sequence=evidence.boundary.terminal_event_sequence,
        )
    except EventQueryResultTooLarge:
        return False
    if latest_completion is None or latest_completion[0] != source.sequence:
        return False
    lifecycle_sequences = [
        record.sequence
        for record in evidence.events
        if record.event.type
        in {
            EventType.SESSION_STARTED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_FORKED,
        }
    ]
    if not lifecycle_sequences or source.sequence <= max(lifecycle_sequences):
        return False
    interaction_lifecycle_ids = {
        record.event.interaction_id
        for record in evidence.lifecycle_events
        if record.event.interaction_id is not None
    }
    return receipt.interaction_id in interaction_lifecycle_ids


async def _latest_conversational_model_completion(
    *,
    service: CayuService,
    session_id: str,
    before_sequence: int,
) -> tuple[int, Event] | None:
    """Find the latest ordinary completion within a strict auxiliary-event bound."""

    remaining = MAX_PRODUCT_RESULT_EVIDENCE_SCAN_EVENTS
    cursor = before_sequence
    while remaining > 0:
        page_size = min(16, remaining)
        records = await service.cayu_app.session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
                before_sequence=cursor,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=page_size,
            )
        )
        if not records:
            return None
        for record in records:
            if is_conversational_model_completion_payload(record.event.payload):
                return record.sequence, record.event
        remaining -= len(records)
        if len(records) < page_size:
            return None
        cursor = records[-1].sequence
    return None


async def _run_product_operation_with_heartbeat(
    *,
    service: CayuService,
    operation: ProductOperation,
    claim_id: str,
    execute_and_settle: Callable[[], Awaitable[_ProductExecutionResult]],
) -> _ProductExecutionResult:
    """Retain authoritative ownership through runtime work and terminal settlement."""

    stop_heartbeat = asyncio.Event()

    async def execute() -> _ProductExecutionResult:
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
            operation=operation,
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
    operation: ProductOperation,
    claim_id: str,
    status: Literal["completed", "failed"],
    result: str | None,
    cancellation: asyncio.CancelledError | None = None,
) -> ProductOperation:
    """Settle one terminal product update before propagating caller cancellation."""

    async def finish() -> ProductOperation:
        finished_operation = await _reconcile_ambiguous_product_store_write(
            lambda: product_store.finish(
                work_id=operation.work_id,
                claim_id=claim_id,
                status=status,
                result=result,
            ),
            operation="product operation finalization",
        )
        return _validated_product_store_operation(
            finished_operation,
            source="finish",
            expected={
                **_product_operation_authority_fields(operation),
                "status": status,
                "result": result,
            },
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


async def _release_product_claim_after_failure(
    *,
    product_store: ProductOperationStore,
    operation: ProductOperation,
    claim_id: str,
    failure: BaseException,
    action: str,
) -> Never:
    """Release failed pending work without hiding either failure or cancellation."""

    try:
        await _release_product_execution_claim_resisting_cancellation(
            product_store,
            work_id=operation.work_id,
            claim_id=claim_id,
        )
    except asyncio.CancelledError as release_cancellation:
        if isinstance(failure, asyncio.CancelledError):
            _raise_cancellation_with_failure(
                failure,
                release_cancellation,
                operation=action,
            )
        _raise_cancellation_with_failure(
            release_cancellation,
            failure,
            operation=action,
        )
    except BaseException as release_error:
        if isinstance(failure, asyncio.CancelledError):
            _raise_cancellation_with_failure(
                failure,
                release_error,
                operation=action,
            )
        raise BaseExceptionGroup(
            f"{action.capitalize()} and execution-claim release both failed.",
            [failure, release_error],
        ) from None
    raise failure


async def _reconcile_ambiguous_product_store_write(
    write: Callable[[], Awaitable[Any]],
    *,
    operation: str,
) -> Any:
    """Retry one idempotent store write with the same identity after lost acknowledgement."""

    try:
        return await write()
    except (
        ProductExecutionClaimLost,
        ProductOperationSettlementConflict,
        ProductResultReceiptConflict,
    ):
        raise
    except asyncio.CancelledError:
        raise
    except Exception as first_failure:
        try:
            return await write()
        except (
            ProductExecutionClaimLost,
            ProductOperationSettlementConflict,
            ProductResultReceiptConflict,
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
            "subject_id": principal.subject_id,
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
                subject_id=principal.subject_id,
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
                {
                    "subject_id": principal.subject_id,
                    "request_text": body.request,
                    "status": "pending",
                    "result": None,
                }
            )
        operation = _validated_product_store_operation(
            reservation.operation,
            source="reserve",
            expected=expected_authority,
        )
        if operation.status == "pending":
            completed = await service._execute_work(
                operation.work_id,
                expected_operation=operation,
            )
            if completed is None:
                raise RuntimeError("Reserved product work disappeared before execution.")
            operation = _validated_product_store_operation(
                completed,
                source="execute_work",
                expected={
                    **_product_operation_authority_fields(operation),
                },
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
        continuation_loop_policy_provider=service._continuation_loop_policies,
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
        recovery_status=operation.recovery_status,
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
