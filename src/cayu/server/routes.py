"""API routes for the cayu server."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, TypeVar, cast
from unicodedata import category as unicode_category
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

from cayu._exception_groups import exception_tree_contains
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    copy_durable_json_object,
    copy_json_value,
    copy_label_map,
    json_utf8_size_within_limit,
    require_clean_nonblank,
    require_durable_json_text,
    require_unicode_scalar_text,
)
from cayu.artifacts import (
    ArtifactListResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.core.events import (
    Event,
    EventType,
    event_durable_sequence,
    event_id_is_runtime_generated,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import Message, MessageRole
from cayu.core.thinking import ThinkingConfig
from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec, eval_corpus_to_json
from cayu.evals.execution import (
    CompiledCorpusSuite,
    CorpusTarget,
    _validate_corpus_target_compatibility,
    compile_corpus_suite,
    evaluation_target_identity,
)
from cayu.evals.execution_comparison import compare_corpus_execution_results
from cayu.evals.execution_reporting import (
    corpus_execution_result_to_json,
    render_corpus_execution_html,
)
from cayu.evals.models import Trajectory
from cayu.evals.promotion import (
    CapturedEvaluationCandidateV1,
    CapturedRunScoreV1,
    PromotionCandidateV1,
    SessionPromotionError,
    build_captured_evaluation_candidate,
    build_promotion_candidate,
    corpus_from_captured_evaluation_candidate,
    corpus_from_promotion_candidate,
    export_captured_evaluation_corpus,
    export_promotion_corpus,
    runnable_promotion_candidate,
    score_captured_evaluation_candidate,
    score_promotion_candidate,
)
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
)
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalBaselineConflict,
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalCaseCatalogPage,
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalCorpusCatalogEntry,
    EvalCorpusCatalogPage,
    EvalCorpusConflict,
    EvalResultConflict,
    EvalResultPage,
    EvalResultQuery,
    EvalRunAdmissionConflict,
    EvalRunCostBudget,
    EvalRunInvocation,
    EvalRunPage,
    EvalRunQuery,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunStatus,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogPage,
    EvalSuiteCatalogQuery,
)
from cayu.evals.trajectory import (
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    trajectory_from_session,
)
from cayu.project_control_plane import ResolvedProjectControlPlaneContext
from cayu.runtime._binding_cleanup import is_containable_cleanup_error
from cayu.runtime._event_projection import (
    PUBLIC_EVENT_ID_PREFIX,
    public_event_id,
    public_event_linkage_id,
    public_event_sequence,
)
from cayu.runtime.aggregates import (
    UsageRollupInconsistent,
    UsageRollupResultTooLarge,
    UsageRollupStoreResult,
    estimate_usage_rollup_cost,
    estimate_usage_session_cost_breakdown,
    summary_usage_metrics_from_event_payload,
)
from cayu.runtime.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.checkpoints import CheckpointCompatibilityError
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    PriceBook,
    SessionCostSummary,
)
from cayu.runtime.costs import (
    estimate_causal_budget_cost as build_causal_budget_cost_summary,
)
from cayu.runtime.costs import (
    estimate_session_cost as build_session_cost_summary,
)
from cayu.runtime.errors import TerminalEventPublicationUncertain
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.runtime.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionSummaryEvidence,
)
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    TaskExecutionSource,
)
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.provider_operations import (
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    copy_provider_operation_resolution_metadata,
    inspect_provider_operation,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import (
    SESSION_MESSAGE_CONTENT_MAX_BYTES,
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    EventRecord,
    InterruptSessionRequest,
    LabelSelectorOperator,
    LabelSelectorRequirement,
    ModelTarget,
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    PendingActionResultTooLarge,
    PendingActionSession,
    ResumeRequest,
    RunRequest,
    Session,
    SessionDebugState,
    SessionMessageDeliveryMode,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionStatus,
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    TerminalSessionEvidenceErrorCode,
    TranscriptQuery,
    UsageRollupQuery,
    _with_runtime_resume_transport_metadata,
    decode_session_cursor,
    decode_session_topology_cursor,
    event_summary_from_records,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_invocation,
    session_outcome_from_records,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    Task,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskTopologyCycle,
    TaskTopologyInconsistent,
    TaskTopologyNode,
    TaskTopologyQuery,
    TaskTopologyStoreResult,
    TaskTopologyTraversalLimitExceeded,
    decode_task_topology_cursor,
    task_create_with_runtime_invocation,
)
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.usage import (
    AggregateUsageMetrics,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    UsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
    causal_budget_usage_summary,
)
from cayu.runtime.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.server._capabilities import inspect_control_plane_capabilities
from cayu.server._diagnostics import inspect_system_diagnostics
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency
from cayu.server.config import EvalsConfig, EvaluationPromotionConfig, normalize_api_path
from cayu.server.contracts import (
    AGGREGATE_ENDPOINT_RESPONSES,
    ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
    ARTIFACT_ENDPOINT_ERROR_RESPONSES,
    BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    CAUSAL_BUDGET_SUMMARY_ENDPOINT_RESPONSES,
    EVALS_ENDPOINT_RESPONSES,
    EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
    MAX_CAPTURED_EVALUATION_REQUEST_BYTES,
    MAX_CONTROL_PLANE_METADATA_BYTES,
    MAX_CONTROL_PLANE_METADATA_MEMBERS,
    MAX_CONTROL_PLANE_METADATA_NESTING,
    MAX_CONTROL_PLANE_PROMPT_BYTES,
    MAX_CONTROL_PLANE_REQUEST_BYTES,
    MAX_EVALS_REQUEST_BYTES,
    MAX_EVALUATION_PROMOTION_REQUEST_BYTES,
    MAX_SESSION_TOPOLOGY_REQUEST_BYTES,
    MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS,
    MAX_USAGE_ROLLUP_REQUEST_BYTES,
    MAX_USAGE_ROLLUP_RESULT_BYTES,
    PENDING_ACTION_ENDPOINT_RESPONSES,
    RUN_ENDPOINT_RESPONSES,
    SERVER_API_PREFIX,
    SESSION_TOPOLOGY_ENDPOINT_RESPONSES,
    STREAMING_ENDPOINT_RESPONSES,
    USAGE_ROLLUP_ENDPOINT_RESPONSES,
    AgentsResponse,
    ApiInteractionSummary,
    ApiReviewedKnowledgeEntry,
    ApiSession,
    ApiSessionDetail,
    ApiTaskDetail,
    ApiTaskListItem,
    ArtifactReadResponse,
    ArtifactsResponse,
    CapturedEvaluationConversion,
    CapturedEvaluationDraft,
    CapturedEvaluationExportRequest,
    CapturedEvaluationLaunchRequest,
    CapturedEvaluationLaunchResponse,
    CapturedEvaluationPreviewRequest,
    CapturedEvaluationPreviewResponse,
    CapturedEvaluationSaveRequest,
    CapturedEvaluationSaveResponse,
    CausalBudgetSummaryResponse,
    ClientGenerationContract,
    EnvironmentsResponse,
    EvalBaselineSelectionRequest,
    EvalBaselineSelectionResponse,
    EvalComparisonRequest,
    EvalComparisonResponse,
    EvalResultDetailResponse,
    EvalResultResponse,
    EvalRunCreateRequest,
    EvalTargetCatalogResponse,
    EvaluationPromotionDraft,
    EvaluationPromotionExportRequest,
    EvaluationPromotionPreviewRequest,
    EvaluationPromotionPreviewResponse,
    HealthResponse,
    ListSessionEventsResponse,
    ListSessionInteractionsResponse,
    ListSessionsResponse,
    OperationalSnapshotRequest,
    OperationalSnapshotResponse,
    PendingActionsResponse,
    PendingKnowledgeDetailResponse,
    PendingKnowledgeListResponse,
    ServerContractResponse,
    SessionsSummaryResponse,
    SessionStateResponse,
    SessionSummaryResponse,
    SessionTopologyRequest,
    SessionTopologyResponse,
    SessionTranscriptResponse,
    SystemDiagnosticsResponse,
    UsageBreakdownItem,
    UsageRollupRequest,
    UsageRollupResponse,
)
from cayu.server.evals_registry import (
    EvalTargetRegistration,
    generated_eval_target_registry,
    resolved_evals_runtime,
    target_for_eval_invocation,
)
from cayu.server.evals_worker import EvalRunCoordinator
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_OBSERVER_MAX_BYTES,
    SSE_OBSERVER_MAX_FRAMES,
    SSE_REPLAY_PAGE_EVENTS,
    SSE_REPLAY_START_MARKER_FORMAT,
    SSE_SEND_TIMEOUT_SECONDS,
    SseErrorCode,
    SseErrorKind,
    SseEventFrameTooLargeError,
    SseObserverLaggedError,
    error_to_sse_message,
    event_to_sse_message,
    parse_last_event_id,
    sse_message_data_bytes,
)
from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListItem,
    KnowledgeReviewWorkflow,
    KnowledgeRevisionConflict,
    KnowledgeVisibility,
)
from cayu.vaults import REDACTED_SECRET

logger = logging.getLogger(__name__)


def _private_no_store_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"Cache-Control": "private, no-store"},
    )


def _private_no_store_validation_error_response(detail: str) -> JSONResponse:
    """Return a bounded validation envelope without reflecting rejected input."""

    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body"],
                    "msg": detail,
                }
            ]
        },
        headers={"Cache-Control": "private, no-store"},
    )


def _parse_json_without_duplicate_keys(body: bytes) -> object:
    """Parse one request body while rejecting non-portable JSON spellings."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number {value!r} is not supported.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object keys are not supported.")
            result[key] = value
        return result

    return json.loads(
        body,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _validated_model_json(value: BaseModel, model_type: type[BaseModel]) -> bytes:
    validated = model_type.model_validate(value.model_dump(mode="python"))
    return validated.model_dump_json().encode("utf-8")


def _render_utf8(renderer: Callable[[Any], str], value: Any) -> bytes:
    return renderer(value).encode("utf-8")


async def _model_json_response(
    value: BaseModel,
    model_type: type[BaseModel],
    *,
    status_code: int = 200,
) -> Response:
    """Serialize a bounded validated response without occupying the server loop."""

    content = await asyncio.to_thread(_validated_model_json, value, model_type)
    return Response(
        content=content,
        media_type="application/json",
        status_code=status_code,
    )


@dataclass(frozen=True)
class _PreparsedPrivateJsonBody:
    """One private JSON body parsed before FastAPI request validation."""

    value: object


_PREPARSED_PRIVATE_JSON_SCOPE_KEY = "cayu.preparsed_private_json_body"
_PrivateBodyModel = TypeVar("_PrivateBodyModel", bound=BaseModel)


async def _validated_private_json_body(
    request: Request,
    model_type: type[_PrivateBodyModel],
    *,
    invalid_detail: str,
) -> _PrivateBodyModel:
    content_type = request.headers.get("content-type")
    if content_type is not None:
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "application/json" and not (
            media_type.startswith("application/") and media_type.endswith("+json")
        ):
            raise HTTPException(status_code=422, detail=invalid_detail)
    parsed_body = request.scope.get(_PREPARSED_PRIVATE_JSON_SCOPE_KEY)
    if not isinstance(parsed_body, _PreparsedPrivateJsonBody):
        raise HTTPException(status_code=422, detail=invalid_detail)
    try:
        return await asyncio.to_thread(model_type.model_validate, parsed_body.value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=422, detail=invalid_detail) from exc


def _json_request_openapi(model: str | type[BaseModel]) -> dict[str, Any]:
    if isinstance(model, str):
        schema = {"$ref": f"#/components/schemas/{model}"}
    else:
        generated = model.model_json_schema()
        definitions = generated.pop("$defs", {})

        def inline_definitions(value: Any, active: frozenset[str] = frozenset()) -> Any:
            if isinstance(value, list):
                return [inline_definitions(item, active) for item in value]
            if not isinstance(value, dict):
                return value
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
                if name in active or name not in definitions:
                    raise ValueError(
                        "Private request OpenAPI schema contains an invalid reference."
                    )
                replacement = inline_definitions(definitions[name], active | {name})
                return {
                    **replacement,
                    **{
                        key: inline_definitions(item, active)
                        for key, item in value.items()
                        if key != "$ref"
                    },
                }
            return {key: inline_definitions(item, active) for key, item in value.items()}

        schema = inline_definitions(generated)
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": schema}},
        }
    }


class _BoundedPrivateJsonBodyRoute(APIRoute):
    """Bound and sanitize a private JSON body before validation can expose input."""

    max_request_bytes: int
    invalid_request_detail: str
    oversized_request_detail: str
    reject_duplicate_json_keys = False
    preparse_auth: AuthDependency | None = None

    def _invalid_request_response(self) -> JSONResponse:
        return _private_no_store_error_response(422, self.invalid_request_detail)

    def get_route_handler(self):
        route_handler = super().get_route_handler()
        max_request_bytes = self.max_request_bytes
        invalid_request_response = self._invalid_request_response
        oversized_request_detail = self.oversized_request_detail
        reject_duplicate_json_keys = self.reject_duplicate_json_keys
        preparse_auth_dependency = (
            None if self.preparse_auth is None else server_auth_dependency(self.preparse_auth)
        )

        async def bounded_route_handler(request: Request) -> Response:
            original_receive = request.receive
            auth_received = bytearray()
            auth_body_complete = False

            if preparse_auth_dependency is not None:

                async def bounded_auth_receive():
                    nonlocal auth_body_complete
                    message = await original_receive()
                    if message["type"] == "http.request":
                        chunk = message.get("body", b"")
                        if len(auth_received) + len(chunk) > max_request_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=oversized_request_detail,
                            )
                        auth_received.extend(chunk)
                        if not message.get("more_body", False):
                            auth_body_complete = True
                    return message

                auth_request = Request(request.scope, receive=bounded_auth_receive)
                try:
                    await preparse_auth_dependency(auth_request)
                except HTTPException as exc:
                    headers = dict(exc.headers or {})
                    headers["Cache-Control"] = "private, no-store"
                    raise HTTPException(
                        status_code=exc.status_code,
                        detail=exc.detail,
                        headers=headers,
                    ) from exc

            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except ValueError:
                    return invalid_request_response()
                if declared_bytes < 0:
                    return invalid_request_response()
                if declared_bytes > max_request_bytes:
                    return _private_no_store_error_response(
                        413,
                        oversized_request_detail,
                    )

            if preparse_auth_dependency is not None:
                received = auth_received
                try:
                    while not auth_body_complete:
                        message = await bounded_auth_receive()
                        if message["type"] != "http.request":
                            return invalid_request_response()
                    raw_body = bytes(received)
                    if reject_duplicate_json_keys and raw_body:
                        parsed = await asyncio.to_thread(
                            _parse_json_without_duplicate_keys,
                            raw_body,
                        )
                        request.scope[_PREPARSED_PRIVATE_JSON_SCOPE_KEY] = (
                            _PreparsedPrivateJsonBody(parsed)
                        )
                except HTTPException as exc:
                    if exc.status_code != 413:
                        raise
                    return _private_no_store_error_response(
                        413,
                        oversized_request_detail,
                    )
                except (UnicodeDecodeError, ValueError, RecursionError):
                    return invalid_request_response()

                replayed = False

                async def replay_receive():
                    nonlocal replayed
                    if replayed:
                        return {"type": "http.disconnect"}
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": raw_body,
                        "more_body": False,
                    }

                bounded_request = Request(request.scope, receive=replay_receive)
            elif reject_duplicate_json_keys:
                received = bytearray()
                try:
                    while True:
                        message = await original_receive()
                        if message["type"] != "http.request":
                            return invalid_request_response()
                        chunk = message.get("body", b"")
                        if len(received) + len(chunk) > max_request_bytes:
                            return _private_no_store_error_response(
                                413,
                                oversized_request_detail,
                            )
                        received.extend(chunk)
                        if not message.get("more_body", False):
                            break
                    raw_body = bytes(received)
                    if raw_body:
                        parsed = await asyncio.to_thread(
                            _parse_json_without_duplicate_keys,
                            raw_body,
                        )
                        request.scope[_PREPARSED_PRIVATE_JSON_SCOPE_KEY] = (
                            _PreparsedPrivateJsonBody(parsed)
                        )
                except (UnicodeDecodeError, ValueError, RecursionError):
                    return invalid_request_response()

                replayed = False

                async def replay_receive():
                    nonlocal replayed
                    if replayed:
                        return {"type": "http.disconnect"}
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": raw_body,
                        "more_body": False,
                    }

                bounded_request = Request(request.scope, receive=replay_receive)
            else:
                received_bytes = 0

                async def bounded_receive():
                    nonlocal received_bytes
                    message = await original_receive()
                    if message["type"] == "http.request":
                        received_bytes += len(message.get("body", b""))
                        if received_bytes > max_request_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=oversized_request_detail,
                            )
                    return message

                bounded_request = Request(request.scope, receive=bounded_receive)
            try:
                response = await route_handler(bounded_request)
            except RequestValidationError:
                return invalid_request_response()
            except HTTPException as exc:
                headers = dict(exc.headers or {})
                headers["Cache-Control"] = "private, no-store"
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=exc.detail,
                    headers=headers,
                ) from exc
            response.headers["Cache-Control"] = "private, no-store"
            return response

        return bounded_route_handler


class _BoundedSessionTopologyRoute(_BoundedPrivateJsonBodyRoute):
    """Bound and sanitize topology bodies before FastAPI exposes validation input."""

    max_request_bytes = MAX_SESSION_TOPOLOGY_REQUEST_BYTES
    invalid_request_detail = "Invalid session topology request."
    oversized_request_detail = "Session topology request exceeds the server byte limit."


class _BoundedControlPlaneRequestRoute(_BoundedPrivateJsonBodyRoute):
    """Bound mutation bodies before validation, provider calls, or durable writes."""

    max_request_bytes = MAX_CONTROL_PLANE_REQUEST_BYTES
    invalid_request_detail = "Invalid control-plane request."
    oversized_request_detail = "Control-plane request exceeds the server byte limit."
    reject_duplicate_json_keys = True


class _BoundedEvaluationPromotionRoute(_BoundedPrivateJsonBodyRoute):
    """Bound runnable-promotion candidates before JSON parsing or validation."""

    max_request_bytes = MAX_EVALUATION_PROMOTION_REQUEST_BYTES
    invalid_request_detail = "Invalid evaluation promotion request."
    oversized_request_detail = "Evaluation promotion request exceeds the server byte limit."
    reject_duplicate_json_keys = True


class _BoundedCapturedEvaluationRoute(_BoundedPrivateJsonBodyRoute):
    """Bound captured-evaluation candidates before JSON parsing or validation."""

    max_request_bytes = MAX_CAPTURED_EVALUATION_REQUEST_BYTES
    invalid_request_detail = "Invalid captured evaluation request."
    oversized_request_detail = "Captured evaluation request exceeds the server byte limit."
    reject_duplicate_json_keys = True


class _BoundedEvalsRoute(_BoundedPrivateJsonBodyRoute):
    """Bound eval corpus and lifecycle bodies before parsing or durable writes."""

    max_request_bytes = MAX_EVALS_REQUEST_BYTES
    invalid_request_detail = "Invalid Evals request."
    oversized_request_detail = "Evals request exceeds the server byte limit."
    reject_duplicate_json_keys = True


def _bounded_evals_route_class(auth: AuthDependency) -> type[_BoundedEvalsRoute]:
    class _AuthenticatedBoundedEvalsRoute(_BoundedEvalsRoute):
        preparse_auth = staticmethod(auth)

    return _AuthenticatedBoundedEvalsRoute


class _BoundedUsageRollupRoute(_BoundedPrivateJsonBodyRoute):
    """Bound and sanitize usage bodies before FastAPI exposes pricing input."""

    max_request_bytes = MAX_USAGE_ROLLUP_REQUEST_BYTES
    invalid_request_detail = "Invalid usage rollup request."
    oversized_request_detail = "Usage rollup request exceeds the server byte limit."

    def _invalid_request_response(self) -> JSONResponse:
        return _private_no_store_validation_error_response(self.invalid_request_detail)


class _ObserverLifecycleEventSourceResponse(EventSourceResponse):
    """Release a waiting detached pump even if response startup fails."""

    def __init__(
        self,
        *args: Any,
        observer_started: asyncio.Event,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._observer_started = observer_started

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._observer_started.set()


_MutationAcceptanceStage = Literal[
    "empty",
    "before_first_event",
    "after_first_event",
    "terminal_uncertainty_acceptance_failed",
    "accepted",
]


@dataclass(frozen=True, slots=True)
class _MutationAcceptanceCallbacks:
    after_first_event: Callable[[Event], Awaitable[None]] | None
    after_terminal_publication_uncertain: (
        Callable[[TerminalEventPublicationUncertain], Awaitable[None]] | None
    )


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PersistableNonBlankString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[^\x00]+$",
    ),
]
ReplaySafeSessionId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*$",
    ),
]
MutationIdHeader = Annotated[
    str | None,
    Header(
        alias="Cayu-Mutation-ID",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*$",
        description=(
            "Client-generated mutation identity used to correlate an ambiguous "
            "SSE reconnect with its durable server acceptance event. Send the same "
            "value on the initial mutation and every Last-Event-ID replay request; "
            "it is a replay correlation key, not permission to repeat the POST."
        ),
    ),
]
ArtifactIdPath = Annotated[str, StringConstraints(min_length=1)]
# Server-entrypoint step budget. The default preserves the historical value while the
# ceiling matches the runtime's own ``max_steps`` bound (RunRequest/ResumeRequest and the
# tool-approval bodies all cap at 256) so a request cannot ask for an unbounded run.
_DEFAULT_RUN_MAX_STEPS = 20
_MAX_RUN_STEPS = 256
_EVENT_PAGE_LIMIT_MAX = 1000
_TRANSCRIPT_PAGE_LIMIT_MAX = 1000
_ARTIFACT_PAGE_LIMIT_MAX = 500
_ARTIFACT_PAGE_OFFSET_MAX = 10_000
_ARTIFACT_CONTENT_BYTES_MAX = 64 * 1024 * 1024
_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES = 512
_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS = 255
_ARTIFACT_ID_HEADER_MAX_CHARS = 512
_ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_KNOWLEDGE_REVIEW_PREVIEW_CHARS = 1200
_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS = 50
_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES = 128_000
_CAUSAL_BUDGET_SUMMARY_MAX_SESSIONS = 500
_CAUSAL_BUDGET_SUMMARY_MAX_EVENTS = 10_000
_CAUSAL_BUDGET_SUMMARY_MAX_EVENT_INPUT_BYTES = 4 * 1024 * 1024
_CAUSAL_BUDGET_SUMMARY_MAX_RESULT_BYTES = 4 * 1024 * 1024
_SERVER_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}
_REPLAY_ACTIVE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
}
_REPLAY_TERMINAL_EVENT_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
_REPLAY_TERMINAL_EVENT_TYPES = frozenset(_REPLAY_TERMINAL_EVENT_BY_STATUS.values())
_REPLAY_TERMINAL_LINEAGE_EVENT_TYPES = {
    EventType.HOOK_STARTED,
    EventType.HOOK_COMPLETED,
    EventType.HOOK_FAILED,
}
_REPLAY_POST_TERMINAL_EVENT_TYPES = {
    EventType.SERVER_MUTATION_ACCEPTED,
    EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
    EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
    EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
}
_REPLAY_OPERATION_START_EVENT_TYPES = {
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.ENVIRONMENT_FACTORY_STARTED,
    EventType.ENVIRONMENT_BINDING_STARTED,
}
# Replays check quickly after reconnect, then back off while a live session is
# quiet so idle streams do not continuously hammer the durable stores.
_REPLAY_POLL_INTERVAL_MIN_S = 0.05
_REPLAY_POLL_INTERVAL_MAX_S = 1.0


def _next_replay_poll_interval(current: float, *, received_events: bool) -> float:
    if received_events:
        return _REPLAY_POLL_INTERVAL_MIN_S
    return min(current * 2, _REPLAY_POLL_INTERVAL_MAX_S)


def _replay_terminal_boundary_id(event: Event) -> str | None:
    """Return the terminal session event represented by an observed record."""
    if event.type in _REPLAY_TERMINAL_EVENT_TYPES:
        return event.id
    if event.type not in _REPLAY_TERMINAL_LINEAGE_EVENT_TYPES:
        return None
    terminal_event_type = event.payload.get("terminal_event_type")
    if terminal_event_type not in _REPLAY_TERMINAL_EVENT_TYPES:
        return None
    terminal_event_id = event.payload.get("terminal_event_id")
    if type(terminal_event_id) is str and terminal_event_id.strip():
        return terminal_event_id
    return None


# Detached event pumps must outlive their SSE consumer (a client disconnect must not
# cancel agent work), so hold strong references until each pump finishes — the event
# loop only keeps weak references to tasks.
_detached_event_pumps: set[asyncio.Task[None]] = set()
_ARTIFACT_SAFE_INLINE_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/csv",
        "text/markdown",
        "text/plain",
    }
)


def _redacted_stream_error_text(cayu_app: Any, error: BaseException) -> str:
    """Return a secret-redacted error string, or a non-sensitive fallback."""
    fallback = f"{type(error).__name__}: stream failed."
    try:
        redacted = cayu_app.redact_json(str(error))
    except Exception:
        return fallback
    if type(redacted) is not str:
        return fallback
    return redacted


def _bounded_run_task_title(redacted_prompt: str) -> str:
    """Apply the 80-character task-title limit without splitting a marker."""

    max_chars = 80
    if len(redacted_prompt) <= max_chars:
        return redacted_prompt
    marker_start = redacted_prompt.rfind(
        REDACTED_SECRET,
        0,
        max_chars + len(REDACTED_SECRET),
    )
    if marker_start >= 0 and marker_start < max_chars < marker_start + len(REDACTED_SECRET):
        return redacted_prompt[:marker_start]
    return redacted_prompt[:max_chars]


def _stream_error_sse_message(
    cayu_app: Any,
    error: BaseException,
    *,
    kind: SseErrorKind,
    code: SseErrorCode,
    retryable: bool,
    session_id: str,
) -> dict[str, str]:
    return error_to_sse_message(
        error,
        kind=kind,
        code=code,
        retryable=retryable,
        session_id=cayu_app.project_session_id_for_exposure(session_id),
        error_text=_redacted_stream_error_text(cayu_app, error),
    )


async def _close_event_stream(event_stream: AsyncIterator[Event]) -> None:
    close = getattr(event_stream, "aclose", None)
    if close is not None:
        # Cleanup is best-effort and must not replace the primary stream outcome.
        # CancelledError is a BaseException, so suppress it explicitly rather than
        # letting a secondary close failure bypass the observer's terminal signal.
        try:
            await close()
        except BaseExceptionGroup as error:
            if not is_containable_cleanup_error(error):
                raise
        except (asyncio.CancelledError, Exception):
            pass


def _preaccept_error_detail(cayu_app: Any, error: BaseException) -> str:
    text = _redacted_stream_error_text(cayu_app, error)
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= SSE_ERROR_TEXT_MAX_BYTES:
        return encoded.decode("utf-8")
    suffix = b"..."
    prefix = encoded[: SSE_ERROR_TEXT_MAX_BYTES - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + suffix.decode()


def _log_mutation_acceptance_failure(
    cayu_app: Any,
    error: BaseException,
    *,
    session_id: str,
    stage: str,
) -> None:
    """Record a bounded, redacted diagnostic for a sanitized HTTP 500."""
    with contextlib.suppress(Exception):
        logger.error(
            "SSE mutation acceptance failed: stage=%s session_id=%r error_type=%s error=%r",
            stage,
            cayu_app.project_session_id_for_exposure(session_id),
            type(error).__name__,
            _preaccept_error_detail(cayu_app, error),
        )


async def _accepted_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
    after_accept: Callable[[Event], Awaitable[None]] | None = None,
    acceptance_callbacks: _MutationAcceptanceCallbacks | None = None,
    conflict_error_types: tuple[type[Exception], ...] = (ValueError,),
) -> EventSourceResponse:
    """Establish one durable event before accepting an SSE mutation response.

    Reconnects are allowed only after the mutation has a durable replay boundary.
    Advancing the runtime here also makes a new session's store insert the atomic
    identity claim before route-owned task creation or an HTTP 200 response.
    """
    if after_accept is not None and acceptance_callbacks is not None:
        raise TypeError("after_accept and acceptance_callbacks are mutually exclusive.")
    first_event_callback = (
        acceptance_callbacks.after_first_event if acceptance_callbacks is not None else after_accept
    )
    loop = asyncio.get_running_loop()
    acceptance: asyncio.Future[tuple[BaseException | None, _MutationAcceptanceStage]] = (
        loop.create_future()
    )
    observer_started = asyncio.Event()

    response, pump_task, abandon_observer = _start_detached_event_stream_response(
        event_stream,
        cayu_app=cayu_app,
        session_id=session_id,
        acceptance=acceptance,
        after_first_event=first_event_callback,
        after_terminal_publication_uncertain=(
            acceptance_callbacks.after_terminal_publication_uncertain
            if acceptance_callbacks is not None
            else None
        ),
        observer_started=observer_started,
    )
    try:
        acceptance_error, stage = await asyncio.shield(acceptance)
    except asyncio.CancelledError:
        # Starting the detached driver transfers runtime ownership away from the
        # request task. A client timeout/disconnect must abandon only its observer;
        # cancelling the driver is reserved for Cayu's interruption protocol.
        abandon_observer()
        raise
    except BaseException:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump_task
        raise

    if acceptance_error is None:
        return response

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await pump_task
    if not isinstance(acceptance_error, Exception):
        raise acceptance_error
    if stage == "terminal_uncertainty_acceptance_failed":
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage=stage,
        )
        raise HTTPException(
            status_code=500,
            detail="Mutation setup failed while recording terminal publication uncertainty.",
        ) from acceptance_error
    if stage == "empty":
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage="before_first_event",
        )
        raise HTTPException(
            status_code=500,
            detail="Mutation completed without producing a durable event.",
        ) from acceptance_error
    if stage == "after_first_event":
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage=stage,
        )
        raise HTTPException(
            status_code=500,
            detail="Mutation setup failed after its durable acceptance event.",
        ) from acceptance_error
    if isinstance(acceptance_error, TerminalEventPublicationUncertain):
        _log_mutation_acceptance_failure(
            cayu_app,
            acceptance_error,
            session_id=session_id,
            stage=stage,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Terminal event publication outcome is uncertain; inspect durable "
                "session state before retrying the mutation."
            ),
        ) from acceptance_error
    if isinstance(acceptance_error, KeyError):
        raise HTTPException(
            status_code=404,
            detail=_preaccept_error_detail(cayu_app, acceptance_error),
        ) from acceptance_error
    if isinstance(acceptance_error, conflict_error_types):
        raise HTTPException(
            status_code=409,
            detail=_preaccept_error_detail(cayu_app, acceptance_error),
        ) from acceptance_error
    _log_mutation_acceptance_failure(
        cayu_app,
        acceptance_error,
        session_id=session_id,
        stage=stage,
    )
    raise HTTPException(
        status_code=500,
        detail="Mutation failed before streaming began.",
    ) from acceptance_error


def _detached_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
) -> EventSourceResponse:
    response, _pump_task, _abandon_observer = _start_detached_event_stream_response(
        event_stream,
        cayu_app=cayu_app,
        session_id=session_id,
    )
    return response


def _start_detached_event_stream_response(
    event_stream: AsyncIterator[Event],
    *,
    cayu_app: Any,
    session_id: str,
    acceptance: asyncio.Future[tuple[BaseException | None, _MutationAcceptanceStage]] | None = None,
    after_first_event: Callable[[Event], Awaitable[None]] | None = None,
    after_terminal_publication_uncertain: (
        Callable[[TerminalEventPublicationUncertain], Awaitable[None]] | None
    ) = None,
    observer_started: asyncio.Event | None = None,
) -> tuple[EventSourceResponse, asyncio.Task[None], Callable[[], None]]:
    """Run ``event_stream`` to completion in a detached task; stream it as an observer.

    The run is driven by the pump task, not by the SSE consumer: a client disconnect
    stops the observer while the session still runs to a terminal state (finalized,
    not stranded RUNNING). Each frame carries a resumable ``id:`` field, and a runtime
    failure surfaces as a terminal structured ``error`` frame instead of an aborted
    connection.
    """
    # Reserve one slot for the terminal done/error signal in addition to the
    # advertised data-frame window. A producer that emits exactly the maximum
    # number of data frames must still be able to close cleanly.
    queue: asyncio.Queue[tuple[str, Any, int]] = asyncio.Queue(maxsize=SSE_OBSERVER_MAX_FRAMES + 1)
    queued_bytes = 0
    queued_event_frames = 0
    observer_accepting = True
    observer_terminal = False

    def discard_queued_items() -> None:
        nonlocal queued_bytes, queued_event_frames
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queued_bytes = 0
        queued_event_frames = 0

    def force_observer_error(item: BaseException) -> None:
        nonlocal observer_terminal
        if not observer_accepting or observer_terminal:
            return
        observer_terminal = True
        discard_queued_items()
        queue.put_nowait(("observer_error", item, 0))

    def abandon_observer() -> None:
        nonlocal observer_accepting
        observer_accepting = False
        discard_queued_items()
        if observer_started is not None:
            observer_started.set()

    def observer_capacity_exceeded(kind: str, data_bytes: int) -> bool:
        return (
            (kind == "event" and queued_event_frames >= SSE_OBSERVER_MAX_FRAMES)
            or queue.full()
            or queued_bytes + data_bytes > SSE_OBSERVER_MAX_BYTES
        )

    async def enqueue(
        kind: str,
        item: Any,
        *,
        data_bytes: int = 0,
        terminal: bool = False,
    ) -> None:
        nonlocal observer_terminal, queued_bytes, queued_event_frames
        if not observer_accepting or observer_terminal:
            return
        if observer_capacity_exceeded(kind, data_bytes):
            # A synchronous event source can produce a full window without an
            # await. Give an already-running observer one scheduling turn to
            # consume queued work before deciding that it has actually lagged.
            # This does not wait for a slow network client or backpressure the
            # detached runtime beyond that single cooperative handoff.
            await asyncio.sleep(0)
            if not observer_accepting or observer_terminal:
                return
        if observer_capacity_exceeded(kind, data_bytes):
            force_observer_error(
                SseObserverLaggedError(
                    "The live observer fell behind its bounded buffer; "
                    "reconnect from the last received event or use durable history."
                )
            )
            return
        queue.put_nowait((kind, item, data_bytes))
        queued_bytes += data_bytes
        if kind == "event":
            queued_event_frames += 1
        if terminal:
            observer_terminal = True

    def resolve_acceptance(
        error: BaseException | None,
        stage: _MutationAcceptanceStage,
    ) -> None:
        if acceptance is not None and not acceptance.done():
            acceptance.set_result((error, stage))

    async def publish_event(event: Event) -> None:
        if not observer_accepting or observer_terminal:
            return
        try:
            sequence = event_durable_sequence(event)
            if sequence is not None and event.id == public_event_id(sequence):
                public_event = event
            elif sequence is None:
                records = await cayu_app.session_store.query_events(
                    EventQuery(
                        session_id=event.session_id,
                        event_id=event.id,
                        limit=2,
                    )
                )
                if len(records) != 1:
                    raise RuntimeError(
                        "Live event has no unique durable record for public projection."
                    )
                record = records[0]
                public_event = cayu_app._project_persisted_event_record_for_exposure(record).event
            else:
                record = EventRecord(sequence=sequence, event=event)
                public_event = cayu_app._project_persisted_event_record_for_exposure(record).event
            message = event_to_sse_message(public_event)
        except SseEventFrameTooLargeError as exc:
            await enqueue("observer_error", exc, terminal=True)
            return
        await enqueue(
            "event",
            message,
            data_bytes=sse_message_data_bytes(message),
        )

    async def pump() -> None:
        saw_first_event = False
        deferred_cancellation: asyncio.CancelledError | None = None

        async def publish_without_losing_cancellation(event: Event) -> None:
            nonlocal deferred_cancellation
            while True:
                try:
                    await publish_event(event)
                    return
                except asyncio.CancelledError as exc:
                    # The runtime iterator is paused at its yield boundary here.
                    # Preserve the cancellation and retry the bounded observer write;
                    # the main loop reissues it only after resuming the iterator.
                    deferred_cancellation = exc

        async def finish_acceptance_bookkeeping(bookkeeping: Awaitable[None]) -> None:
            """Finish acceptance bookkeeping without swallowing an interrupt."""
            nonlocal deferred_cancellation
            callback_task = asyncio.ensure_future(bookkeeping)
            while True:
                try:
                    await asyncio.shield(callback_task)
                except asyncio.CancelledError as exc:
                    if callback_task.cancelled():
                        raise RuntimeError(
                            "Mutation acceptance bookkeeping was cancelled."
                        ) from exc
                    # The detached pump owns this task, so cancellation can arrive
                    # while route-owned bookkeeping is in progress. Let the
                    # one-shot write finish; the caller redelivers cancellation
                    # only after its acceptance boundary is durable.
                    deferred_cancellation = exc
                    continue
                return

        try:
            while True:
                if deferred_cancellation is not None:
                    current_task = asyncio.current_task()
                    if current_task is None:
                        raise deferred_cancellation
                    # Scheduling (rather than immediately throwing) lets anext()
                    # resume through wrapper yield boundaries first. Cancellation is
                    # then delivered at the runtime's next real await (factory,
                    # binding, provider stream, or interruption status check).
                    asyncio.get_running_loop().call_soon(current_task.cancel)
                    deferred_cancellation = None
                try:
                    event = await anext(event_stream)
                except StopAsyncIteration:
                    if not saw_first_event:
                        resolve_acceptance(
                            RuntimeError("Mutation completed without producing a durable event."),
                            "empty",
                        )
                    break
                except TerminalEventPublicationUncertain as exc:
                    if saw_first_event or after_terminal_publication_uncertain is None:
                        raise
                    try:
                        await finish_acceptance_bookkeeping(
                            after_terminal_publication_uncertain(exc)
                        )
                    except BaseException as callback_error:
                        resolve_acceptance(
                            callback_error,
                            "terminal_uncertainty_acceptance_failed",
                        )
                        raise
                    resolve_acceptance(None, "accepted")
                    await _close_event_stream(event_stream)
                    await enqueue("runtime_error", exc, terminal=True)
                    if deferred_cancellation is not None:
                        raise deferred_cancellation from None
                    return

                is_first_event = not saw_first_event
                saw_first_event = True
                if is_first_event and acceptance is not None:
                    try:
                        if after_first_event is not None:
                            await finish_acceptance_bookkeeping(after_first_event(event))
                    except BaseException as exc:
                        resolve_acceptance(exc, "after_first_event")
                        await _close_event_stream(event_stream)
                        raise
                    else:
                        resolve_acceptance(None, "accepted")

                await publish_without_losing_cancellation(event)

                if (
                    is_first_event
                    and observer_started is not None
                    and deferred_cancellation is None
                ):
                    try:
                        await observer_started.wait()
                    except asyncio.CancelledError as exc:
                        deferred_cancellation = exc
        except asyncio.CancelledError as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await enqueue("done", None, terminal=True)
            raise
        except BaseExceptionGroup as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await _close_event_stream(event_stream)
            if exception_tree_contains(exc, Exception) and is_containable_cleanup_error(exc):
                await enqueue("runtime_error", exc, terminal=True)
            else:
                await enqueue("done", None, terminal=True)
                raise
        except Exception as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await _close_event_stream(event_stream)
            await enqueue("runtime_error", exc, terminal=True)
        except BaseException as exc:
            if not saw_first_event:
                resolve_acceptance(exc, "before_first_event")
            await _close_event_stream(event_stream)
            await enqueue("done", None, terminal=True)
            raise
        else:
            await enqueue("done", None, terminal=True)

    pump_task = asyncio.create_task(pump())
    _detached_event_pumps.add(pump_task)
    pump_task.add_done_callback(_detached_event_pumps.discard)

    async def observe() -> AsyncIterator[dict[str, str]]:
        nonlocal observer_accepting, queued_bytes, queued_event_frames
        if observer_started is not None:
            observer_started.set()
        try:
            while True:
                kind, item, data_bytes = await queue.get()
                queued_bytes -= data_bytes
                if kind == "event":
                    queued_event_frames -= 1
                    yield item
                    continue
                if kind == "runtime_error":
                    publication_uncertain = isinstance(item, TerminalEventPublicationUncertain)
                    yield _stream_error_sse_message(
                        cayu_app,
                        item,
                        kind="runtime",
                        code=(
                            "terminal_event_publication_uncertain"
                            if publication_uncertain
                            else "runtime_failed"
                        ),
                        retryable=publication_uncertain,
                        session_id=session_id,
                    )
                    return
                if kind == "observer_error":
                    frame_too_large = isinstance(item, SseEventFrameTooLargeError)
                    yield _stream_error_sse_message(
                        cayu_app,
                        item,
                        kind="observer",
                        code=("event_frame_too_large" if frame_too_large else "observer_lagged"),
                        retryable=not frame_too_large,
                        session_id=session_id,
                    )
                    return
                return
        finally:
            abandon_observer()

    if observer_started is None:
        response = EventSourceResponse(observe(), send_timeout=SSE_SEND_TIMEOUT_SECONDS)
    else:
        response = _ObserverLifecycleEventSourceResponse(
            observe(),
            send_timeout=SSE_SEND_TIMEOUT_SECONDS,
            observer_started=observer_started,
        )
    return response, pump_task, abandon_observer


class _BoundedControlPlanePromptBody(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    @field_validator("prompt", check_fields=False)
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        require_durable_json_text(value, "prompt")
        if len(value.encode("utf-8")) > MAX_CONTROL_PLANE_PROMPT_BYTES:
            raise ValueError(
                "prompt exceeds the maximum encoded size of "
                f"{MAX_CONTROL_PLANE_PROMPT_BYTES} bytes."
            )
        return value


class RunBody(_BoundedControlPlanePromptBody):
    prompt: NonBlankString
    session_id: ReplaySafeSessionId | None = Field(
        default=None,
        description=(
            "Optional caller-selected identity for replay-safe run observation. "
            f"Before the first event, reconnect with `{SSE_REPLAY_START_MARKER_FORMAT}` "
            "using this session ID."
        ),
    )
    agent: NonBlankString = "assistant"
    target: ModelTarget | None = None
    causal_budget_id: NonBlankString | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_disallowed_fields(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "invocation_origin" in value:
            raise ValueError("invocation_origin is server-owned at the HTTP run boundary.")
        if isinstance(value, Mapping) and "model" in value:
            raise ValueError("model was removed; use target with provider_name and model.")
        return value

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)


class ExecutionProfileAdoptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    idempotency_key: PersistableNonBlankString = Field(max_length=256)
    reason: PersistableNonBlankString = Field(max_length=4096)
    requested_by: ResolutionActor | None = None


class ResumeBody(_BoundedControlPlanePromptBody):
    session_id: NonBlankString
    prompt: NonBlankString
    profile_adoption: ExecutionProfileAdoptionBody | None = None
    max_steps: StrictInt = Field(default=_DEFAULT_RUN_MAX_STEPS, ge=1, le=_MAX_RUN_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)


def _copy_control_plane_metadata(value: Any) -> dict[str, Any]:
    copied = copy_durable_json_object(value, "metadata")
    if not json_utf8_size_within_limit(copied, MAX_CONTROL_PLANE_METADATA_BYTES):
        raise ValueError(
            "metadata exceeds the maximum encoded size of "
            f"{MAX_CONTROL_PLANE_METADATA_BYTES} bytes."
        )

    member_count = 0
    pending: list[tuple[dict[str, Any] | list[Any], int]] = [(copied, 1)]
    while pending:
        container, depth = pending.pop()
        if depth > MAX_CONTROL_PLANE_METADATA_NESTING:
            raise ValueError(
                "metadata exceeds the maximum nesting depth of "
                f"{MAX_CONTROL_PLANE_METADATA_NESTING}."
            )
        member_count += len(container)
        if member_count > MAX_CONTROL_PLANE_METADATA_MEMBERS:
            raise ValueError(
                "metadata exceeds the maximum aggregate member count of "
                f"{MAX_CONTROL_PLANE_METADATA_MEMBERS}."
            )
        values = container.values() if type(container) is dict else container
        for item in values:
            if type(item) is dict or type(item) is list:
                pending.append((item, depth + 1))
    return copied


class _BoundedControlPlaneMetadataBody(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    @field_validator("metadata", mode="before", check_fields=False)
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _copy_control_plane_metadata(value)


class InterruptSessionBody(_BoundedControlPlaneMetadataBody):
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by: ResolutionActor | None = None


class CompactSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: PersistableNonBlankString = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_transcript_cursor: StrictInt = Field(ge=0)
    instructions: NonBlankString | None = Field(default=None, max_length=4096)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    requested_by: ResolutionActor | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> CompactSessionBody:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "CompactSessionBody",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "CompactSessionBody.requested_by",
            )
        return self


class EnqueueSessionMessageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: NonBlankString = Field(max_length=256)
    content: NonBlankString = Field(max_length=SESSION_MESSAGE_CONTENT_MAX_BYTES)
    delivery_mode: SessionMessageDeliveryMode
    requested_by: ResolutionActor | None = None

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, value: str) -> str:
        require_durable_json_text(value, "content")
        if len(value.encode("utf-8")) > SESSION_MESSAGE_CONTENT_MAX_BYTES:
            raise ValueError(
                "content exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return value

    @model_validator(mode="after")
    def validate_durable_text(self) -> EnqueueSessionMessageBody:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "EnqueueSessionMessageBody",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "EnqueueSessionMessageBody.requested_by",
            )
        return self


class UpdateSessionLabelsBody(BaseModel):
    # Required + extra="forbid": a missing/typo'd key must 422, never silently replace
    # all labels with {} (these are full-replacement mutations).
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str]


class UpdateSessionMetadataBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(
        description=(
            "Complete replacement for user-authored session metadata. Cayu-owned "
            "entries are preserved by the store and must not be supplied."
        )
    )


class SessionCostBody(BaseModel):
    pricing: PriceBook
    currency: NonBlankString = "USD"


class SessionsSummaryBody(BaseModel):
    pricing: PriceBook | None = None
    currency: NonBlankString = "USD"


class TaskHoldBody(BaseModel):
    reason: NonBlankString | None = None
    payload: dict[str, Any] | None = None


class ToolApprovalBody(_BoundedControlPlaneMetadataBody):
    """Body for resolving a pending tool approval.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values are accepted only when
    they preserve the invocation's frozen execution profile.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    tool_round_id: NonBlankString
    tool_call_id: NonBlankString
    decision: ToolApprovalDecision
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class ProviderOperationResolutionBody(_BoundedControlPlaneMetadataBody):
    """Explicit retry-or-fail disposition for unavailable provider work."""

    session_id: NonBlankString
    stage_id: NonBlankString = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    action: ProviderOperationResolutionAction
    reason: NonBlankString | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        bounded = _copy_control_plane_metadata(value)
        return copy_provider_operation_resolution_metadata(bounded)


class ToolApprovalRecoveryBody(_BoundedControlPlaneMetadataBody):
    """Body for recovering an approved tool call with an unknown result.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``: the resumed run inherits the original run's configuration
    persisted on the pending approval. Explicit values are accepted only when
    they preserve the invocation's frozen execution profile.
    """

    session_id: NonBlankString
    approval_id: NonBlankString
    tool_round_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class ToolRoundRecoveryBody(_BoundedControlPlaneMetadataBody):
    """Body for recovering a crashed ordinary tool call with an operator outcome.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``, which restores the configuration persisted with the pending
    tool round. An explicit value is accepted only when it preserves the
    invocation's frozen execution profile.
    """

    session_id: NonBlankString
    round_id: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class UserInputResolveBody(_BoundedControlPlaneMetadataBody):
    """Body for answering a session paused by ``ask_user``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    Explicit values are accepted only when they preserve the invocation's frozen execution
    profile.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


class UserInputRecoveryBody(_BoundedControlPlaneMetadataBody):
    """Body for recovering a user-input round stuck on ``manual_recovery_required``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``: the
    resumed run inherits the original run's configuration persisted on the pending user input.
    Explicit values are accepted only when they preserve the invocation's frozen execution
    profile.
    """

    session_id: NonBlankString
    input_id: NonBlankString
    answer: NonBlankString
    tool_call_id: NonBlankString
    outcome: ToolApprovalRecoveryOutcome
    message: NonBlankString
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: NonBlankString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


def _request_actor(
    auth_context: AuthContext | None,
    body_actor: ResolutionActor | None,
    *,
    field_name: Literal["requested_by", "resolved_by"],
) -> ResolutionActor | None:
    """Derive a typed operator actor for an authenticated control-plane route.

    With auth configured, provenance comes from the verified caller and a
    body-supplied actor is rejected loudly (mirroring the reserved ``cayu:``
    label rejection) — a silent override would let clients believe they
    recorded an actor the audit trail replaced. Open-access bodies are accepted
    but re-stamped ``source="request"``, so a request can never claim
    server-verified (``http_auth``) or system provenance.
    """
    if auth_context is not None:
        if body_actor is not None:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} is derived from the authenticated caller and "
                "cannot be supplied in the request body.",
            )
        try:
            return ResolutionActor(
                subject=auth_context.subject,
                tenant=auth_context.tenant,
                source=ResolutionActorSource.HTTP_AUTH,
                claims=auth_context.claims,
            )
        except ValueError as exc:
            # AuthContext.subject is unconstrained, so an auth backend can hand
            # back a reserved ``cayu:``-prefixed subject; surface that as a
            # clean 400 instead of an unhandled 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body_actor is None:
        return None
    try:
        return ResolutionActor(
            subject=body_actor.subject,
            tenant=body_actor.tenant,
            source=ResolutionActorSource.REQUEST,
            claims=body_actor.claims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _request_resolution_actor(
    auth_context: AuthContext | None,
    body_resolved_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_resolved_by, field_name="resolved_by")


def _request_interruption_actor(
    auth_context: AuthContext | None,
    body_requested_by: ResolutionActor | None,
) -> ResolutionActor | None:
    return _request_actor(auth_context, body_requested_by, field_name="requested_by")


def _serialize_event_record(cayu_app: Any, record: EventRecord) -> dict[str, Any]:
    public_record = cayu_app._project_persisted_event_record_for_exposure(record)
    event = public_record.event
    # The event-domain projector is authoritative. A second generic redaction
    # pass would corrupt validated controls and sequence aliases under short
    # secret collisions.
    return copy_json_value(
        {
            "sequence": public_record.sequence,
            "id": event.id,
            "type": str(event.type),
            "session_id": event.session_id,
            "interaction_id": event.interaction_id,
            "agent_name": event.agent_name,
            "environment_name": event.environment_name,
            "workflow_name": event.workflow_name,
            "tool_name": event.tool_name,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat(),
        },
        "event",
    )


def _serialize_pending_action(
    cayu_app: Any,
    action: PendingActionRecord,
) -> dict[str, Any]:
    """Project an event-derived action without publishing its private discriminator."""

    payload = _redact_control_plane_values(
        cayu_app,
        action.model_dump(
            mode="json",
            exclude={"session", "event"},
        ),
        "pending_action",
        preserve_string_fields={"kind", "policy_evidence"},
        untrusted_container_fields={"arguments"},
    )
    payload["id"] = f"{public_event_id(action.event.sequence)}:{action.kind.value}"
    for response_field, event_field in (
        ("approval_id", "approval_id"),
        ("input_id", "input_id"),
        ("round_id", "tool_round_id"),
        ("tool_call_id", "tool_call_id"),
    ):
        private_value = getattr(action, response_field)
        if private_value is None:
            payload[response_field] = None
            continue
        durable_value = action.source_linkage.get(event_field)
        if durable_value is None:
            # Checkpoint-derived recovery state can lack schema-owned linkage in
            # its bounded source event. It remains visible for diagnosis, but is
            # intentionally non-actionable rather than exposing private linkage.
            payload[response_field] = None
            continue
        if durable_value != private_value:
            raise RuntimeError(
                f"Pending action {response_field} disagrees with its durable event authority."
            )
        payload[response_field] = public_event_linkage_id(action.event.sequence, event_field)
    payload["session"] = _serialize_session_base(cayu_app, action.session)
    payload["event"] = _serialize_event_record(cayu_app, action.event)
    return payload


def _serialize_interaction_record(cayu_app: Any, record: EventRecord) -> dict[str, Any]:
    public_record = cayu_app._project_persisted_event_record_for_exposure(record)
    event = public_record.event
    interaction_id = event.interaction_id
    if interaction_id is None:
        raise RuntimeError("Interaction lifecycle event has no interaction identity.")
    evidence = InteractionSummaryEvidence.model_validate(event.payload)
    terminal = event.type in INTERACTION_TERMINAL_EVENT_TYPES
    return copy_json_value(
        {
            "interaction_id": interaction_id,
            "session_id": event.session_id,
            **evidence.model_dump(mode="json"),
            "start_event_sequence": evidence.start_event_sequence or public_record.sequence,
            "terminal_event_id": event.id if terminal else None,
            "terminal_event_sequence": public_record.sequence if terminal else None,
            "updated_at": event.timestamp.isoformat(),
        },
        "interaction",
    )


def _serialize_session_outcome(cayu_app: Any, outcome: SessionOutcome) -> dict[str, Any]:
    return {
        "session_id": cayu_app.project_session_id_for_exposure(outcome.session_id),
        "status": outcome.status.value,
        "reason": _redact_control_plane_json(
            cayu_app,
            outcome.reason,
            "session_outcome.reason",
        ),
        "details": _redact_control_plane_json(
            cayu_app,
            outcome.details,
            "session_outcome.details",
        ),
        "retry": _redact_control_plane_json(
            cayu_app,
            outcome.retry,
            "session_outcome.retry",
        ),
        "terminal_event": (
            None
            if outcome.terminal_event is None
            else _serialize_event_record(cayu_app, outcome.terminal_event)
        ),
        "latest_retry_event": (
            None
            if outcome.latest_retry_event is None
            else _serialize_event_record(cayu_app, outcome.latest_retry_event)
        ),
    }


def _serialize_session_base(
    cayu_app: Any,
    session: Session | PendingActionSession,
) -> dict[str, Any]:
    # Shared list-view fields. The list endpoint omits the (potentially large,
    # unbounded) per-session metadata; callers fetch a single session to get it.
    payload = _redact_control_plane_values(
        cayu_app,
        {
            "id": session.id,
            "status": session.status.value,
            "agent_name": session.agent_name,
            "provider_name": session.provider_name,
            "model": session.model,
            "parent_session_id": session.parent_session_id,
            "causal_budget_id": session.causal_budget_id,
            "runtime_name": session.runtime_name,
            "runtime_version": session.runtime_version,
            "environment_name": session.environment_name,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "labels": session.labels,
        },
        "session",
        preserve_string_fields={"created_at", "status", "updated_at"},
        untrusted_container_fields={"labels"},
    )
    payload["id"] = cayu_app.project_session_id_for_exposure(session.id)
    payload["parent_session_id"] = (
        None
        if session.parent_session_id is None
        else cayu_app.project_session_id_for_exposure(session.parent_session_id)
    )
    payload["causal_budget_id"] = cayu_app.project_causal_budget_id_for_exposure(
        session.causal_budget_id,
        session_ids=(
            session.id,
            *(() if session.parent_session_id is None else (session.parent_session_id,)),
        ),
    )
    return payload


def _serialize_causal_budget_usage_summary(
    cayu_app: Any,
    summary: CausalBudgetUsageSummary,
) -> dict[str, Any]:
    payload = summary.model_dump()
    payload["causal_budget_id"] = cayu_app.project_causal_budget_id_for_exposure(
        summary.causal_budget_id,
        session_ids=summary.session_ids,
    )
    payload["session_ids"] = [
        cayu_app.project_session_id_for_exposure(session_id) for session_id in summary.session_ids
    ]
    payload["session_summaries"] = [
        _serialize_session_usage_summary(cayu_app, session_summary)
        for session_summary in summary.session_summaries
    ]
    return payload


def _serialize_session_usage_summary(
    cayu_app: Any,
    summary: SessionUsageSummary,
) -> dict[str, Any]:
    return {
        **summary.model_dump(),
        "session_id": cayu_app.project_session_id_for_exposure(summary.session_id),
    }


def _serialize_causal_budget_cost_summary(
    cayu_app: Any,
    summary: CausalBudgetCostSummary,
) -> dict[str, Any]:
    payload = summary.model_dump(mode="json")
    payload["causal_budget_id"] = cayu_app.project_causal_budget_id_for_exposure(
        summary.causal_budget_id,
        session_ids=summary.session_ids,
    )
    payload["session_ids"] = [
        cayu_app.project_session_id_for_exposure(session_id) for session_id in summary.session_ids
    ]
    payload["session_costs"] = [
        _serialize_session_cost_summary(cayu_app, session_cost)
        for session_cost in summary.session_costs
    ]
    return payload


def _serialize_session_cost_summary(
    cayu_app: Any,
    summary: SessionCostSummary,
) -> dict[str, Any]:
    return {
        **summary.model_dump(mode="json"),
        "session_id": cayu_app.project_session_id_for_exposure(summary.session_id),
    }


def _serialize_session(cayu_app: Any, session: Session) -> dict[str, Any]:
    return {
        **_serialize_session_base(cayu_app, session),
        "metadata": _redact_control_plane_json(
            cayu_app,
            session.metadata,
            "session.metadata",
        ),
    }


def _serialize_session_detail(cayu_app: Any, session: Session) -> dict[str, Any]:
    return {
        **_serialize_session(cayu_app, session),
        "invocation": {
            "schema_version": session.invocation.schema_version,
            "origin": {
                "trust": session.invocation.origin.trust.value,
                "subject": _redact_control_plane_json(
                    cayu_app,
                    session.invocation.origin.subject,
                    "session.invocation.origin.subject",
                ),
                "tenant": _redact_control_plane_json(
                    cayu_app,
                    session.invocation.origin.tenant,
                    "session.invocation.origin.tenant",
                ),
            },
            "root_invocation_id": session.invocation.root_invocation_id,
            "root_session_id": cayu_app.project_session_id_for_exposure(
                session.invocation.root_session_id
            ),
            "source": session.invocation.source.value,
        },
    }


def _serialize_session_topology_node(
    cayu_app: Any,
    node: SessionTopologyNode,
) -> dict[str, Any]:
    return _redact_control_plane_values(
        cayu_app,
        {
            "id": node.id,
            "agent_name": node.agent_name,
            "provider_name": node.provider_name,
            "model": node.model,
            "parent_session_id": node.parent_session_id,
            "causal_budget_id": node.causal_budget_id,
            "runtime_name": node.runtime_name,
            "runtime_version": node.runtime_version,
            "environment_name": node.environment_name,
            "status": node.status.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "last_activity_at": node.last_activity_at.isoformat(),
        },
        "session_topology.node",
        preserve_string_fields={
            "created_at",
            "last_activity_at",
            "status",
            "updated_at",
        },
    )


def _require_safe_session_topology_authority(
    cayu_app: Any,
    result: SessionTopologyStoreResult,
) -> None:
    """Fail closed when redaction would corrupt graph identity or linkage."""

    nodes = (
        result.focus,
        *result.ancestors,
        *result.expanded_parents,
        *(child for branch in result.branches for child in branch.children),
    )
    structural_values: set[str] = set()
    for node in nodes:
        structural_values.add(node.id)
        structural_values.add(node.causal_budget_id)
        if node.parent_session_id is not None:
            structural_values.add(node.parent_session_id)
    structural_values.update(branch.parent_session_id for branch in result.branches)
    for value in structural_values:
        redacted = _redact_control_plane_json(
            cayu_app,
            value,
            "session_topology.authority",
        )
        if type(redacted) is not str or redacted != value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Session topology identity cannot cross the configured redaction boundary."
                ),
            )


def _require_safe_usage_session_authority(
    cayu_app: Any,
    result: UsageRollupStoreResult,
) -> None:
    """Fail closed when redaction would corrupt per-session usage identity."""

    if result.session_breakdown is None:
        return
    for group in result.session_breakdown.groups:
        redacted = _redact_control_plane_json(
            cayu_app,
            group.session_id,
            "usage_rollup.session_id",
        )
        if type(redacted) is not str or redacted != group.session_id:
            raise HTTPException(
                status_code=409,
                detail=("Usage session identity cannot cross the configured redaction boundary."),
                headers={"Cache-Control": "private, no-store"},
            )


def _serialize_session_cursor(cayu_app: Any, cursor: str | None) -> str | None:
    """Return a pagination cursor only when its keyset authority is secret-free."""

    if cursor is None:
        return None
    redacted_cursor = _redact_control_plane_json(
        cayu_app,
        cursor,
        "session_cursor",
    )
    if type(redacted_cursor) is not str or redacted_cursor != cursor:
        raise HTTPException(
            status_code=409,
            detail=(
                "Session pagination cannot continue because its cursor authority "
                "contains a configured workload secret."
            ),
        )
    try:
        _, session_id = decode_session_cursor(cursor)
    except ValueError:
        # SessionStore cursors are intentionally opaque. Built-in stores use
        # encode_session_cursor(), which lets us inspect embedded authority,
        # while custom stores may return any secret-free portable token.
        return cursor
    redacted_session_id = _redact_control_plane_json(
        cayu_app,
        session_id,
        "session_cursor.session_id",
    )
    if type(redacted_session_id) is not str or redacted_session_id != session_id:
        # Redacting the ID inside the cursor would change its keyset position and
        # silently skip or duplicate sessions. Reject the legacy authority until
        # the server has a configured opaque cursor codec.
        raise HTTPException(
            status_code=409,
            detail=(
                "Session pagination cannot continue because its cursor authority "
                "contains a configured workload secret."
            ),
        )
    return cursor


def _serialize_session_topology_cursor(
    cayu_app: Any,
    cursor: str | None,
    *,
    parent_session_id: str,
) -> str | None:
    """Project a parent-bound topology cursor only when its authority is safe."""

    if cursor is None:
        return None
    redacted_cursor = _redact_control_plane_json(
        cayu_app,
        cursor,
        "session_topology.cursor",
    )
    if type(redacted_cursor) is not str or redacted_cursor != cursor:
        raise HTTPException(
            status_code=409,
            detail=(
                "Session topology cannot continue because its cursor authority "
                "contains a configured workload secret."
            ),
        )
    try:
        _, child_session_id = decode_session_topology_cursor(
            cursor,
            parent_session_id=parent_session_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail="The session store returned an invalid topology cursor.",
        ) from exc
    for field_name, value in (
        ("parent_session_id", parent_session_id),
        ("child_session_id", child_session_id),
    ):
        redacted_value = _redact_control_plane_json(
            cayu_app,
            value,
            f"session_topology.cursor.{field_name}",
        )
        if type(redacted_value) is not str or redacted_value != value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Session topology cannot continue because its cursor authority "
                    "contains a configured workload secret."
                ),
            )
    return cursor


def _task_topology_nodes(
    result: TaskTopologyStoreResult,
) -> tuple[TaskTopologyNode, ...]:
    nodes_by_id: dict[str, TaskTopologyNode] = {}
    for node in (
        *result.expanded_parents,
        *(task for branch in result.session_branches for task in branch.tasks),
        *(task for branch in result.child_branches for task in branch.children),
    ):
        nodes_by_id.setdefault(node.id, node)
    return tuple(nodes_by_id.values())


def _require_safe_task_topology_authority(
    cayu_app: Any,
    result: TaskTopologyStoreResult,
) -> None:
    """Fail closed when redaction would corrupt task identity or linkage."""

    structural_values: set[str] = set()
    for node in _task_topology_nodes(result):
        structural_values.add(node.id)
        if node.session_id is not None:
            structural_values.add(node.session_id)
        if node.parent_task_id is not None:
            structural_values.add(node.parent_task_id)
    structural_values.update(branch.session_id for branch in result.session_branches)
    structural_values.update(branch.parent_task_id for branch in result.child_branches)
    for value in structural_values:
        redacted = _redact_control_plane_json(
            cayu_app,
            value,
            "task_topology.authority",
        )
        if type(redacted) is not str or redacted != value:
            raise HTTPException(
                status_code=409,
                detail=("Task topology identity cannot cross the configured redaction boundary."),
            )


def _serialize_task_topology_node(
    cayu_app: Any,
    node: TaskTopologyNode,
) -> dict[str, Any]:
    serialized = _redact_control_plane_values(
        cayu_app,
        {
            "id": node.id,
            "type": node.type,
            "title": node.title,
            "status": node.status.value,
            "status_reason": node.status_reason,
            "session_id": node.session_id,
            "parent_task_id": node.parent_task_id,
            "assigned_agent_name": node.assigned_agent_name,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
        },
        "task_topology.node",
        preserve_string_fields={"created_at", "status", "updated_at"},
    )
    truncated_fields = set(node.truncated_fields)
    for field_name in (
        "type",
        "title",
        "assigned_agent_name",
        "status_reason",
    ):
        value = serialized[field_name]
        if value is not None and len(value.encode("utf-8")) > (
            TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES
        ):
            # Replacing a short secret with the redaction marker can enlarge a
            # field beyond the store projection's bound. Omit it truthfully
            # rather than letting post-redaction output escape that ceiling.
            serialized[field_name] = None
            truncated_fields.add(field_name)
    canonical_fields = ("type", "title", "assigned_agent_name", "status_reason")
    serialized["truncated_fields"] = [
        field_name for field_name in canonical_fields if field_name in truncated_fields
    ]
    return serialized


def _serialize_task_topology_cursor(
    cayu_app: Any,
    cursor: str | None,
    *,
    scope_kind: Literal["session", "parent_task"],
    scope_id: str,
) -> str | None:
    if cursor is None:
        return None
    redacted_cursor = _redact_control_plane_json(
        cayu_app,
        cursor,
        "task_topology.cursor",
    )
    if type(redacted_cursor) is not str or redacted_cursor != cursor:
        raise HTTPException(
            status_code=409,
            detail=(
                "Task topology cannot continue because its cursor authority "
                "contains a configured workload secret."
            ),
        )
    try:
        _, task_id = decode_task_topology_cursor(
            cursor,
            scope_kind=scope_kind,
            scope_id=scope_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail="The task store returned an invalid topology cursor.",
        ) from exc
    for field_name, value in (("scope_id", scope_id), ("task_id", task_id)):
        redacted = _redact_control_plane_json(
            cayu_app,
            value,
            f"task_topology.cursor.{field_name}",
        )
        if type(redacted) is not str or redacted != value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Task topology cannot continue because its cursor authority "
                    "contains a configured workload secret."
                ),
            )
    return cursor


def _serialize_task_topology_projection(
    cayu_app: Any,
    result: TaskTopologyStoreResult | None,
    *,
    status: Literal["available", "not_configured", "unsupported"],
) -> dict[str, Any]:
    if result is None:
        return {
            "status": status,
            "observed_at": None,
            "session_branches": [],
            "expanded_parents": [],
            "child_branches": [],
            "unique_node_count": 0,
        }
    return {
        "status": "available",
        "observed_at": result.observed_at,
        "session_branches": [
            {
                "session_id": branch.session_id,
                "tasks": [_serialize_task_topology_node(cayu_app, task) for task in branch.tasks],
                "next_cursor": _serialize_task_topology_cursor(
                    cayu_app,
                    branch.next_cursor,
                    scope_kind="session",
                    scope_id=branch.session_id,
                ),
                "has_more": branch.has_more,
            }
            for branch in result.session_branches
        ],
        "expanded_parents": [
            _serialize_task_topology_node(cayu_app, node) for node in result.expanded_parents
        ],
        "child_branches": [
            {
                "parent_task_id": branch.parent_task_id,
                "children": [
                    _serialize_task_topology_node(cayu_app, child) for child in branch.children
                ],
                "next_cursor": _serialize_task_topology_cursor(
                    cayu_app,
                    branch.next_cursor,
                    scope_kind="parent_task",
                    scope_id=branch.parent_task_id,
                ),
                "has_more": branch.has_more,
            }
            for branch in result.child_branches
        ],
        "unique_node_count": len(_task_topology_nodes(result)),
    }


def _execution_topology_edges(
    session_result: SessionTopologyStoreResult,
    task_result: TaskTopologyStoreResult | None,
) -> list[dict[str, Any]]:
    session_nodes_by_id: dict[str, SessionTopologyNode] = {}
    for node in (
        session_result.focus,
        *session_result.ancestors,
        *session_result.expanded_parents,
        *(child for branch in session_result.branches for child in branch.children),
    ):
        session_nodes_by_id.setdefault(node.id, node)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def append_edge(
        kind: Literal["session_parent", "task_parent", "task_session"],
        source_id: str,
        target_id: str,
        *,
        target_loaded: bool,
    ) -> None:
        key = (kind, source_id, target_id)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "kind": kind,
                "source_id": source_id,
                "target_id": target_id,
                "target_loaded": target_loaded,
            }
        )

    loaded_session_ids = set(session_nodes_by_id)
    for node in session_nodes_by_id.values():
        if node.parent_session_id is not None:
            append_edge(
                "session_parent",
                node.id,
                node.parent_session_id,
                target_loaded=node.parent_session_id in loaded_session_ids,
            )

    if task_result is not None:
        task_nodes = _task_topology_nodes(task_result)
        loaded_task_ids = {node.id for node in task_nodes}
        for node in task_nodes:
            if node.parent_task_id is not None:
                append_edge(
                    "task_parent",
                    node.id,
                    node.parent_task_id,
                    target_loaded=node.parent_task_id in loaded_task_ids,
                )
            if node.session_id is not None:
                append_edge(
                    "task_session",
                    node.id,
                    node.session_id,
                    target_loaded=node.session_id in loaded_session_ids,
                )
    return edges


def _redact_control_plane_json(cayu_app: Any, value: Any, field_name: str) -> Any:
    copied = copy_json_value(value, field_name)
    redactor = getattr(cayu_app, "redact_json", None)
    if callable(redactor):
        return redactor(copied)
    return copied


def _redact_control_plane_values(
    cayu_app: Any,
    value: Any,
    field_name: str,
    *,
    preserve_string_fields: set[str] | frozenset[str] = frozenset(),
    untrusted_container_fields: set[str] | frozenset[str] = frozenset(),
) -> Any:
    """Redact values recursively without rewriting typed response keys."""

    copied = copy_json_value(value, field_name)
    redactor = getattr(cayu_app, "redact_json", None)
    if not callable(redactor):
        return copied
    redact_json = cast("Callable[[Any], Any]", redactor)

    def redact(item: Any, *, item_field: str | None = None) -> Any:
        if type(item) is str:
            if item_field in preserve_string_fields:
                return item
            return redact_json(item)
        if type(item) is list:
            return [redact(child, item_field=item_field) for child in item]
        if type(item) is dict:
            return {
                key: (
                    redact_json(child)
                    if key in untrusted_container_fields
                    else redact(child, item_field=key)
                )
                for key, child in item.items()
            }
        return item

    return redact(copied)


def _serialize_tool(cayu_app: Any, tool: Any) -> dict[str, Any]:
    effect = getattr(tool.effect, "value", str(tool.effect))
    return {
        "name": tool.name,
        "description": _redact_control_plane_json(cayu_app, tool.description, "description"),
        "input_schema": _redact_control_plane_json(cayu_app, tool.schema, "input_schema"),
        "parallel_safe": tool.parallel_safe,
        "effect": effect,
        "workspace_mutation": tool.workspace_mutation,
    }


def _serialize_agent(cayu_app: Any, agent: Any) -> dict[str, Any]:
    spec = agent.spec
    thinking = (
        None
        if spec.thinking is None
        else _redact_control_plane_json(cayu_app, spec.thinking.model_dump(mode="json"), "thinking")
    )
    tools = [_serialize_tool(cayu_app, tool) for tool in agent.tools.values()]
    return {
        "name": spec.name,
        "provider_name": spec.provider_name,
        "model": spec.model,
        "tool_count": len(tools),
        "tools": sorted(tools, key=lambda item: item["name"]),
        "metadata": _redact_control_plane_json(cayu_app, spec.metadata, "metadata"),
        "provider_options": _redact_control_plane_json(
            cayu_app,
            spec.provider_options,
            "provider_options",
        ),
        "thinking": thinking,
        "has_system_prompt": spec.system_prompt is not None and bool(spec.system_prompt.strip()),
    }


def _object_type_name(value: Any) -> str | None:
    if value is None:
        return None
    return type(value).__name__


def _object_id(value: Any) -> str | None:
    if value is None:
        return None
    object_id = getattr(value, "id", None)
    return object_id if isinstance(object_id, str) and object_id.strip() else None


def _workspace_instruction_summary(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return "inline"
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return "inline"
    mode = getattr(value, "mode", None)
    if isinstance(mode, str):
        return mode
    return type(value).__name__


def _serialize_environment(cayu_app: Any, record: Any) -> dict[str, Any]:
    environment = record.environment
    workspace = environment.workspace
    artifact_store = environment.artifact_store
    bound_workspace = record.bound_workspace
    bound_payload = None
    if bound_workspace is not None:
        bound_payload = {
            "source_workspace_id": _object_id(bound_workspace.source_workspace),
            "bound_workspace_id": _object_id(bound_workspace.workspace),
            "runner_type": _object_type_name(bound_workspace.runner),
            "path": bound_workspace.path,
            "metadata": _redact_control_plane_json(
                cayu_app,
                bound_workspace.metadata,
                "metadata",
            ),
        }
    return {
        "name": record.spec.name,
        "metadata": _redact_control_plane_json(cayu_app, record.spec.metadata, "metadata"),
        "is_factory": record.factory is not None,
        "workspace_id": _object_id(workspace),
        "artifact_store_id": _object_id(artifact_store),
        "runner_type": _object_type_name(environment.runner),
        "binding_type": _object_type_name(environment.binding),
        "vault_type": _object_type_name(environment.vault),
        "proxy_type": _object_type_name(environment.proxy),
        "knowledge_store_type": _object_type_name(environment.knowledge_store),
        "mcp_server_count": len(environment.mcp_servers),
        "workspace_instructions": _workspace_instruction_summary(
            environment.workspace_instructions
        ),
        "bound_workspace": bound_payload,
    }


def _artifact_stores_by_id(cayu_app: Any) -> dict[str, ArtifactStore]:
    stores: dict[str, ArtifactStore] = {}
    for record in cayu_app.list_environment_registrations():
        store = record.environment.artifact_store
        if isinstance(store, ArtifactStore):
            try:
                store_id = require_clean_nonblank(store.id, "artifact_store.id")
                store_id = require_unicode_scalar_text(store_id, "artifact_store.id")
            except ValueError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="An artifact store has an invalid id configuration.",
                ) from exc
            existing = stores.get(store_id)
            if existing is not None and existing is not store:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Multiple registered environments use the same artifact_store_id: "
                        f"{store_id}. Configure unique artifact store ids."
                    ),
                )
            stores[store_id] = store
    return stores


def _serialize_artifact(cayu_app: Any, metadata: Any, *, artifact_store_id: str) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "artifact_store_id": artifact_store_id,
        "filename": metadata.filename,
        "content_type": metadata.content_type,
        "size_bytes": metadata.size_bytes,
        "scope": metadata.scope.value,
        "session_id": metadata.session_id,
        "agent_name": metadata.agent_name,
        "environment_name": metadata.environment_name,
        "created_at": metadata.created_at.isoformat(),
        "metadata": _redact_control_plane_json(cayu_app, metadata.metadata, "metadata"),
    }


def _artifact_sort_key(artifact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(artifact["created_at"]),
        str(artifact["artifact_store_id"]),
        str(artifact["id"]),
    )


def _decode_artifact_text(content: bytes, content_type: str) -> str | None:
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type.startswith("text/") or normalized_content_type in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace")
    return None


def _artifact_read_preview(cayu_app: Any, read: Any) -> tuple[str, str | None]:
    text_preview = _decode_artifact_text(read.content, read.metadata.content_type)
    if text_preview is None:
        return base64.b64encode(read.content).decode("ascii"), None
    redacted_preview = _redact_control_plane_json(cayu_app, text_preview, "artifact.content")
    if not isinstance(redacted_preview, str):
        raise TypeError("Artifact text preview redaction must return a string.")
    return base64.b64encode(redacted_preview.encode("utf-8")).decode("ascii"), redacted_preview


def _artifact_content_disposition(filename: str, disposition: str) -> str:
    safe_filename = "".join(
        "_"
        if char in {"/", "\\"}
        or unicode_category(char) in _ARTIFACT_UNSAFE_FILENAME_UNICODE_CATEGORIES
        else char
        for char in filename
    ).strip()
    if not safe_filename:
        safe_filename = "artifact"
    safe_filename = _truncate_utf8_filename(
        safe_filename,
        max_bytes=_ARTIFACT_FILENAME_HEADER_UTF8_MAX_BYTES,
    )
    ascii_filename = "".join(
        char if 0x20 <= ord(char) < 0x7F and char not in {'"', "/", "\\"} else "_"
        for char in safe_filename
    ).strip()
    if not ascii_filename:
        ascii_filename = "artifact"
    ascii_filename = ascii_filename[:_ARTIFACT_FILENAME_HEADER_ASCII_MAX_CHARS]
    encoded_filename = quote(safe_filename, safe="", encoding="utf-8", errors="replace")
    return f"{disposition}; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"


def _truncate_utf8_filename(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    stem, separator, extension = value.rpartition(".")
    if not separator:
        stem = value
    suffix = f".{extension}" if separator and 0 < len(extension) <= 32 else ""
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        suffix = ""
        suffix_bytes = b""
        stem = value
    prefix_bytes = stem.encode("utf-8")[: max_bytes - len(suffix_bytes)]
    prefix = prefix_bytes.decode("utf-8", errors="ignore")
    return f"{prefix}{suffix}" or "artifact"


def _artifact_content_disposition_kind(content_type: str, requested: str) -> str:
    if requested != "inline":
        return "attachment"
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type in _ARTIFACT_SAFE_INLINE_CONTENT_TYPES:
        return "inline"
    return "attachment"


def _artifact_header_value(value: str, fallback: str) -> str:
    for candidate in (value, fallback, "unknown"):
        stripped = candidate.strip()
        if stripped and all(0x20 <= ord(char) < 0x7F for char in stripped):
            return stripped[:_ARTIFACT_ID_HEADER_MAX_CHARS]
    return "unknown"


def _usage_breakdown(
    events: list[Event],
    *,
    key_fn: Callable[[UsageMetrics], tuple[str | None, str | None]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for event in events:
        if event.type != EventType.MODEL_COMPLETED:
            continue
        try:
            metrics = summary_usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            continue
        if metrics is None:
            continue
        provider_name, model = key_fn(metrics)
        key = (provider_name, model)
        bucket = buckets.setdefault(
            key,
            {
                "provider_name": provider_name,
                "model": model,
                "session_ids": set(),
                "model_steps": 0,
                "usage": build_aggregate_usage_metrics(),
            },
        )
        bucket["session_ids"].add(event.session_id)
        bucket["model_steps"] += 1
        bucket["usage"] = _add_usage_metrics(bucket["usage"], metrics)

    items = [
        UsageBreakdownItem(
            provider_name=provider_name,
            model=model,
            session_count=len(bucket["session_ids"]),
            model_steps=bucket["model_steps"],
            usage=bucket["usage"],
        ).model_dump()
        for (provider_name, model), bucket in buckets.items()
    ]
    return sorted(
        items,
        key=lambda item: (
            -item["usage"]["total_tokens"],
            item["provider_name"] or "",
            item["model"] or "",
        ),
    )


def _add_usage_metrics(
    left: AggregateUsageMetrics | UsageMetrics,
    right: UsageMetrics,
) -> AggregateUsageMetrics:
    if isinstance(left, UsageMetrics):
        aggregate = add_aggregate_usage(build_aggregate_usage_metrics(), left)
    else:
        aggregate = left
    return add_aggregate_usage(aggregate, right)


def _parse_session_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Session label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate session label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_session_label_selectors(
    values: list[str] | None,
) -> tuple[LabelSelectorRequirement, ...]:
    if values is None:
        return ()
    selectors: list[LabelSelectorRequirement] = []
    for raw in values:
        if type(raw) is not str:
            raise HTTPException(status_code=422, detail="Label selector must be a string.")
        for expression in _split_label_selector(raw):
            selectors.append(_parse_label_selector_expression(expression))
    return tuple(selectors)


def _split_label_selector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if not part:
                raise HTTPException(status_code=422, detail="Invalid label selector.")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    part = value[start:].strip()
    if not part:
        raise HTTPException(status_code=422, detail="Invalid label selector.")
    parts.append(part)
    return parts


def _parse_label_selector_expression(expression: str) -> LabelSelectorRequirement:
    try:
        if expression.startswith("!"):
            key = expression[1:].strip()
            return LabelSelectorRequirement(
                key=key,
                operator=LabelSelectorOperator.NOT_EXISTS,
            )
        if " notin " in expression:
            key, raw_values = expression.split(" notin ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=_parse_label_selector_values(raw_values),
            )
        if " in " in expression:
            key, raw_values = expression.split(" in ", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=_parse_label_selector_values(raw_values),
            )
        if "!=" in expression:
            key, value = expression.split("!=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.NOT_IN,
                values=(value.strip(),),
            )
        if "==" in expression:
            key, value = expression.split("==", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        if "=" in expression:
            key, value = expression.split("=", 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=LabelSelectorOperator.IN,
                values=(value.strip(),),
            )
        return LabelSelectorRequirement(
            key=expression.strip(),
            operator=LabelSelectorOperator.EXISTS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_label_selector_values(raw_values: str) -> tuple[str, ...]:
    raw_values = raw_values.strip()
    if not raw_values.startswith("(") or not raw_values.endswith(")"):
        raise HTTPException(status_code=422, detail="Label selector values must use `(a,b)`.")
    values = tuple(value.strip() for value in raw_values[1:-1].split(","))
    if any(not value for value in values):
        raise HTTPException(status_code=422, detail="Label selector values cannot be blank.")
    return values


def _clean_optional_query_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        return require_clean_nonblank(value, field_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _trace_context_metadata(http_request: Request) -> dict[str, Any]:
    # Carry an inbound W3C trace context into the session metadata so an
    # OpenTelemetryEventSink can root the session span under the caller's trace.
    # Used as a shared dependency by every route that starts a traced session.
    metadata: dict[str, Any] = {}
    traceparent = http_request.headers.get("traceparent")
    if traceparent:
        metadata["traceparent"] = traceparent
        tracestate = http_request.headers.get("tracestate")
        if tracestate:
            metadata["tracestate"] = tracestate
    return metadata


TraceContextMetadata = Annotated[dict[str, Any], Depends(_trace_context_metadata)]


def _serialize_message_part(cayu_app: Any, part: Any) -> dict[str, Any]:
    excluded_fields = {
        field_name
        for field_name in ("model_step_id", "model_attempt_id", "tool_round_id")
        if getattr(part, field_name, None) is None
    }
    if part.type == "thinking":
        # The opaque round-trip state (Anthropic signatures / redacted blobs) is
        # provider-internal and must not be exposed to transcript API consumers.
        excluded_fields.add("provider_state")
    payload = part.model_dump(mode="json", exclude=excluded_fields)
    return _redact_control_plane_values(
        cayu_app,
        payload,
        "transcript_message.part",
        # These values belong to validated file-attachment protocol structure,
        # not caller data. Preserve them so redaction cannot turn a valid
        # attachment into an uninterpretable shape when a short secret happens
        # to overlap "image", "document", or a supported MIME type.
        preserve_string_fields={"content_type", "kind", "type"},
        untrusted_container_fields={
            "arguments",
            "artifacts",
            "metadata",
            "provider_state",
            "state",
            "structured",
        },
    )


def _serialize_transcript_message(
    cayu_app: Any,
    session_id: str,
    index: int,
    message: Message,
    interaction_id: str | None,
) -> dict[str, Any]:
    return {
        "index": index,
        "interaction_id": (
            None
            if interaction_id is None
            else cayu_app.project_interaction_id_for_exposure(
                interaction_id,
                session_id=session_id,
            )
        ),
        "role": str(message.role),
        "content": [_serialize_message_part(cayu_app, part) for part in message.content],
    }


def _serialize_task_list_item(cayu_app: Any, task: Task) -> dict[str, Any]:
    projected = _redact_control_plane_values(
        cayu_app,
        {
            "id": task.id,
            "type": task.type,
            "title": task.title,
            "description": task.description,
            "status": task.status.value,
            "status_reason": task.status_reason,
            "status_payload": task.status_payload,
            "session_id": task.session_id,
            "parent_task_id": task.parent_task_id,
            "assigned_agent_name": task.assigned_agent_name,
            "available_at": task.available_at.isoformat() if task.available_at else None,
            "worker_id": task.worker_id,
            "lease_expires_at": (
                task.lease_expires_at.isoformat() if task.lease_expires_at else None
            ),
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        },
        "task",
        preserve_string_fields={
            "available_at",
            "completed_at",
            "created_at",
            "lease_expires_at",
            "status",
            "updated_at",
        },
        untrusted_container_fields={"status_payload"},
    )
    projected_status_payload = projected.get("status_payload")
    if (
        task.retry_series is not None
        and type(task.status_payload) is dict
        and type(projected_status_payload) is dict
        and type(task.status_payload.get("cost_currency")) is str
    ):
        projected_status_payload["cost_currency"] = cayu_app.redact_uppercase_text(
            task.status_payload["cost_currency"]
        )
    return {
        **projected,
        "retry_series": (
            None if task.retry_series is None else _serialize_task_retry_series(cayu_app, task)
        ),
    }


def _serialize_task_retry_series(cayu_app: Any, task: Task) -> dict[str, Any]:
    series = task.retry_series
    if series is None:
        raise AssertionError("Task retry-series serialization requires retry authority.")
    projected = series.model_dump(
        mode="json",
        warnings=False,
        exclude={"authority_sha256"},
    )
    projected["causal_budget_id"] = cayu_app.project_causal_budget_id_for_exposure(
        series.causal_budget_id,
        session_ids=(),
    )
    projected["cumulative_tokens"] = str(series.cumulative_tokens)
    projected["tokens_remaining"] = (
        None if series.tokens_remaining is None else str(series.tokens_remaining)
    )
    projected_policy = projected["policy"]
    if type(projected_policy) is not dict:
        raise AssertionError("Task retry policy serialization returned a non-object.")
    projected_policy["max_total_tokens"] = (
        None if series.policy.max_total_tokens is None else str(series.policy.max_total_tokens)
    )
    projected_policy["cost_currency"] = cayu_app.redact_uppercase_text(series.policy.cost_currency)
    return _redact_control_plane_values(
        cayu_app,
        projected,
        "task.retry_series",
        preserve_string_fields={
            "cumulative_estimated_cost",
            "cumulative_tokens",
            "disposition",
            "elapsed_deadline",
            "estimated_cost_remaining",
            "max_estimated_cost",
            "max_total_tokens",
            "next_eligible_at",
            "started_at",
            "tokens_remaining",
        },
    )


def _serialize_task_detail(cayu_app: Any, task: Task) -> dict[str, Any]:
    return {
        **_serialize_task_list_item(cayu_app, task),
        "invocation": {
            "schema_version": task.invocation.schema_version,
            "origin": {
                "trust": task.invocation.origin.trust.value,
                "subject": _redact_control_plane_json(
                    cayu_app,
                    task.invocation.origin.subject,
                    "task.invocation.origin.subject",
                ),
                "tenant": _redact_control_plane_json(
                    cayu_app,
                    task.invocation.origin.tenant,
                    "task.invocation.origin.tenant",
                ),
            },
            "root_invocation_id": task.invocation.root_invocation_id,
            "root_session_id": (
                None
                if task.invocation.root_session_id is None
                else cayu_app.project_session_id_for_exposure(task.invocation.root_session_id)
            ),
            "source": task.invocation.source.value,
        },
        "input": _redact_control_plane_json(cayu_app, task.input, "task.input"),
        "result": _redact_control_plane_json(cayu_app, task.result, "task.result"),
        "error": _redact_control_plane_json(cayu_app, task.error, "task.error"),
        "metadata": _redact_control_plane_json(
            cayu_app,
            task.metadata,
            "task.metadata",
        ),
        "started_at": task.started_at.isoformat() if task.started_at else None,
    }


def _serialize_knowledge_entry_base(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.id,
        "revision": entry.revision,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "visibility": entry.visibility.value,
        "status": entry.status.value,
        "title": entry.title,
        "labels": dict(entry.labels),
        "aspects": list(entry.aspects),
        "impact_targets": list(entry.impact_targets),
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "created_by_type": entry.created_by_type.value,
        "created_by": entry.created_by,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "importance": entry.importance,
        "importance_source": entry.importance_source,
        "confidence": entry.confidence,
    }


def _serialize_knowledge_list_item(item: KnowledgeListItem) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(item.entry),
        "chunk_count": item.chunk_count,
        "text_preview": item.text_preview,
    }


def _serialize_reviewed_knowledge_entry(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text_preview": _knowledge_text_preview(entry.text),
    }


def _serialize_knowledge_detail(entry: KnowledgeEntry) -> dict[str, Any]:
    return {
        **_serialize_knowledge_entry_base(entry),
        "text": entry.text,
        "metadata": entry.metadata,
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
    }


def _serialize_knowledge_chunk(chunk: KnowledgeChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "entry_id": chunk.entry_id,
        "entry_revision": chunk.entry_revision,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_uri": chunk.source_uri,
        "metadata": dict(chunk.metadata),
    }


def _knowledge_text_preview(text: str) -> str:
    if len(text) <= _KNOWLEDGE_REVIEW_PREVIEW_CHARS:
        return text
    return f"{text[:_KNOWLEDGE_REVIEW_PREVIEW_CHARS]}..."


def _parse_knowledge_label_filters(values: list[str] | None) -> dict[str, str]:
    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise HTTPException(
                status_code=422,
                detail="Knowledge label filters must use `key=value`.",
            )
        key, value = raw.split("=", 1)
        try:
            parsed = copy_label_map({key: value}, "label")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_key, parsed_value = next(iter(parsed.items()))
        if parsed_key in labels:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate knowledge label filter: {parsed_key}",
            )
        labels[parsed_key] = parsed_value
    return labels


def _parse_knowledge_string_filters(values: list[str] | None, field_name: str) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        try:
            result.append(require_clean_nonblank(value, field_name))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return list(dict.fromkeys(result))


def create_router(
    *,
    cayu_app,
    session_store,
    task_store,
    knowledge_store=None,
    knowledge_access_scope=None,
    knowledge_review_namespace: str | None = None,
    knowledge_review_labels: dict[str, str] | None = None,
    auth: AuthDependency | None = None,
    api_path: str = SERVER_API_PREFIX,
    openapi_url: str | None = "/openapi.json",
    replay_idle_timeout_s: float = 300.0,
    dashboard_configured: bool = False,
    dashboard_pricing_configured: bool = False,
    deployment_name: str | None = None,
    dashboard_access_authenticated: bool | None = None,
    docs_enabled: bool | None = None,
    dashboard_pricing_metadata: tuple[str, str] | None = None,
    evaluation_promotion: EvaluationPromotionConfig | None = None,
    evaluation_promotion_pricing: PriceBook | None = None,
    generated_evals_pricing: PriceBook | None = None,
    evals: EvalsConfig | None = None,
    continuation_loop_policy_provider: (
        Callable[[str], Awaitable[tuple[LoopPolicy, ...]]] | None
    ) = None,
    _project_context: ResolvedProjectControlPlaneContext | None = None,
) -> APIRouter:
    """Create an APIRouter with standard cayu endpoints.

    Args:
        auth: FastAPI-compatible dependency guarding the CAYU control plane.
            ``create_server`` callers normally supply this through
            ``AuthenticatedAccess``; only deliberate ``OpenAccess`` should leave
            it unset. It protects every control-plane route that can start, change,
            inspect, or reveal runtime state; only the health route stays open for
            load balancers. It must return ``AuthContext`` or a compatible mapping
            and raise ``HTTPException`` (401/403) to deny a request. Authentication
            does not add tenant-level authorization or storage isolation:
            ``AuthContext.tenant`` is operator provenance only.
        api_path: URL path prefix for the CAYU control plane. Defaults to
            ``/api``.
        openapi_url: Public OpenAPI schema URL advertised by ``/contract`` for
            client generation. Pass ``None`` when generated OpenAPI is disabled.
        replay_idle_timeout_s: Maximum time an active replay stream may wait
            without seeing a new persisted event before emitting an error and closing.
        dashboard_pricing_configured: Whether resolved dashboard configuration
            supplies a default price book for cost estimation. This is discovery
            metadata only; the usage endpoint still validates every submitted
            price book.
        dashboard_configured: Whether the bundled dashboard is mounted. Its own
            configured access dependency remains authoritative.
        deployment_name: Optional resolved server deployment identity. Embedded
            mounts that do not own a ``ServerConfig`` leave it unset.
        dashboard_access_authenticated: Whether the mounted dashboard has an
            authentication dependency. Defaults to the API access posture when
            the dashboard is configured.
        docs_enabled: Whether Cayu owns enabled generated documentation routes.
            Embedded mounts leave this unknown.
        dashboard_pricing_metadata: Validated default catalog version and opaque
            generation provenance. Values that exceed diagnostic bounds are
            omitted from the response.
        evaluation_promotion: Complete authenticated captured-session promotion
            policy. When absent, no promotion route or enabled capability exists.
        evaluation_promotion_pricing: Optional already-validated dashboard price
            book reused for captured cost evidence.
        generated_evals_pricing: Optional server-owned dashboard price book used
            by generated execution targets for cost assertions and run ceilings.
        evals: Complete authenticated durable execution wiring for one explicit
            V1 corpus target. It takes indivisible precedence over generated
            project targets. When both it and project authority are absent, no
            durable Evals route, worker, or enabled capability exists.
        continuation_loop_policy_provider: Optional internal-session policy
            resolver used by an embedding application for resume, approval,
            user-input, and tool-recovery continuations. The returned policies
            remain process-local and are never accepted from HTTP payloads.
    """

    if continuation_loop_policy_provider is not None and not callable(
        continuation_loop_policy_provider
    ):
        raise TypeError("continuation_loop_policy_provider must be callable or None.")
    if (
        _project_context is not None
        and type(_project_context) is not ResolvedProjectControlPlaneContext
    ):
        raise TypeError("_project_context must be framework-owned project context or None.")
    trusted_local_evals_access = (
        _project_context is not None and _project_context.trusted_local_development
    )

    if (
        isinstance(replay_idle_timeout_s, bool)
        or not isinstance(replay_idle_timeout_s, (int, float))
        or not isfinite(replay_idle_timeout_s)
        or replay_idle_timeout_s <= 0
    ):
        raise ValueError("replay_idle_timeout_s must be a finite positive number.")
    replay_idle_timeout_s = float(replay_idle_timeout_s)

    if evaluation_promotion is not None:
        if type(evaluation_promotion) is not EvaluationPromotionConfig:
            raise TypeError("evaluation_promotion must be an EvaluationPromotionConfig or None.")
        if auth is None and not trusted_local_evals_access:
            raise ValueError("evaluation_promotion requires authenticated API access.")
        if evaluation_promotion.source_agent_name not in cayu_app.list_agents():
            raise ValueError("evaluation_promotion.source_agent_name is not registered.")
        promotion_identity = {
            "target_key": evaluation_promotion.target_key,
            "source_agent_name": evaluation_promotion.source_agent_name,
            "application_release_id": evaluation_promotion.application_release_id,
        }
        try:
            redacted_promotion_identity = cayu_app.redact_json(promotion_identity)
        except Exception as exc:
            raise ValueError(
                "evaluation_promotion identity could not cross the application redaction boundary."
            ) from exc
        if redacted_promotion_identity != promotion_identity:
            raise ValueError("evaluation_promotion identity contains a workload secret.")
    if (
        evaluation_promotion_pricing is not None
        and type(evaluation_promotion_pricing) is not PriceBook
    ):
        raise TypeError("evaluation_promotion_pricing must be a PriceBook or None.")
    if evaluation_promotion is None and evaluation_promotion_pricing is not None:
        raise ValueError("evaluation_promotion_pricing requires evaluation_promotion.")
    if generated_evals_pricing is not None and type(generated_evals_pricing) is not PriceBook:
        raise TypeError("generated_evals_pricing must be a PriceBook or None.")

    if evals is not None:
        if type(evals) is not EvalsConfig:
            raise TypeError("evals must be an exact EvalsConfig or None.")
        try:
            evals = EvalsConfig(
                target=evals.target,
                store=evals.store,
                lease_seconds=evals.lease_seconds,
                poll_interval_seconds=evals.poll_interval_seconds,
                shutdown_grace_seconds=evals.shutdown_grace_seconds,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("evals configuration is invalid.") from exc
        if auth is None and not trusted_local_evals_access:
            raise ValueError("evals requires authenticated API access.")
        if evals.target.app is not cayu_app:
            raise ValueError("evals.target must reference the attached CayuApp instance.")
        try:
            evaluation_target_identity(evals.target)
        except Exception as exc:
            raise ValueError("evals target identity is unavailable.") from exc

    automatic_evals_access = auth is not None or trusted_local_evals_access
    generated_registry = None
    if (
        evals is None
        and automatic_evals_access
        and _project_context is not None
        and _project_context.project_id is not None
    ):
        generated_registry = generated_eval_target_registry(
            cayu_app,
            project_id=_project_context.project_id,
            application_release_id=_project_context.application_release_id,
            app_manifest_fingerprint=_project_context.app_manifest_fingerprint,
            app_manifest_project_root=_project_context.app_manifest_project_root,
            price_book=generated_evals_pricing,
        )
    eval_runtime = resolved_evals_runtime(
        explicit=evals,
        registry=generated_registry,
        automatic_store=(
            None
            if not automatic_evals_access or _project_context is None
            else _project_context.eval_store
        ),
    )
    eval_registry = eval_runtime.registry if eval_runtime is not None else generated_registry

    api_prefix = normalize_api_path(api_path, field_name="api_path")

    @contextlib.asynccontextmanager
    async def evals_lifespan(_app):
        coordinator = None if eval_runtime is None else EvalRunCoordinator(eval_runtime)
        if coordinator is not None:
            coordinator.start()
        try:
            yield
        finally:
            if coordinator is not None:
                await coordinator.stop()

    router = APIRouter(prefix=api_prefix, lifespan=evals_lifespan)
    bounded_control_plane_router = APIRouter(route_class=_BoundedControlPlaneRequestRoute)
    bounded_evaluation_promotion_router = APIRouter(route_class=_BoundedEvaluationPromotionRoute)
    bounded_captured_evaluation_router = APIRouter(route_class=_BoundedCapturedEvaluationRoute)
    bounded_evals_router = APIRouter(
        route_class=(_BoundedEvalsRoute if auth is None else _bounded_evals_route_class(auth))
    )
    auth_context_openapi_schema = AuthContext.model_json_schema()
    capability_snapshot = inspect_control_plane_capabilities(
        dashboard_configured=dashboard_configured,
        tasks_configured=task_store is not None,
        knowledge_configured=knowledge_store is not None,
        dashboard_pricing_configured=dashboard_pricing_configured,
        session_usage_aggregates_supported=session_store.supports_usage_aggregates,
        session_topology_supported=session_store.supports_session_topology,
        evaluation_promotion_configured=evaluation_promotion is not None,
        terminal_session_evidence_supported=session_store.supports_terminal_session_evidence,
        session_lineage_supported=session_store.supports_session_lineage,
        evals_configured=eval_runtime is not None,
        eval_store_configured=(
            eval_runtime is not None
            or (_project_context is not None and _project_context.eval_store is not None)
        ),
        eval_target_configured=eval_registry is not None,
        eval_project_identity_configured=(
            (eval_runtime is not None and evals is not None)
            or (_project_context is not None and _project_context.project_id is not None)
        ),
        eval_captured_results_supported=(
            eval_runtime is not None and eval_runtime.store.captured_results
        ),
    )
    if dashboard_access_authenticated is None and dashboard_configured:
        dashboard_access_authenticated = auth is not None
    diagnostics_snapshot = inspect_system_diagnostics(
        capabilities=capability_snapshot,
        deployment_name=deployment_name,
        api_authenticated=auth is not None,
        dashboard_authenticated=dashboard_access_authenticated,
        dashboard_enabled=dashboard_configured,
        docs_enabled=docs_enabled,
        pricing_configured=dashboard_pricing_configured,
        pricing_metadata=dashboard_pricing_metadata,
    )

    # Shared dependency list for control-plane routes. FastAPI treats an empty
    # sequence like no dependencies, so `auth=None` keeps explicit open access.
    auth_dependency = server_auth_dependency(auth) if auth is not None else None
    protected: list[Any] = [Depends(auth_dependency)] if auth_dependency is not None else []

    async def _optional_auth_context(request: Request) -> AuthContext | None:
        # The interruption and approval/user-input resolution routes take this as a
        # handler parameter INSTEAD of `dependencies=protected`: one callable
        # both guards the route (its 401/403 raises before the handler body)
        # and yields the verified caller identity for typed operator
        # provenance. Splitting guard and extraction into two differently
        # wrapped callables would invoke the user's auth dependency twice per
        # request (FastAPI caches per-callable, not per-underlying-auth).
        # Handlers must take this via the `= Depends(...)` default form: with
        # `from __future__ import annotations`, a function-local Annotated
        # alias is an unresolvable string annotation that FastAPI silently
        # degrades to a required query parameter.
        if auth_dependency is None:
            return None
        return await auth_dependency(request)

    optional_auth_context = Depends(_optional_auth_context)

    async def _resolve_public_session_id(value: str) -> str:
        try:
            return await cayu_app._resolve_public_session_id(value)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def _continuation_loop_policies(session_id: str) -> tuple[LoopPolicy, ...]:
        if continuation_loop_policy_provider is None:
            return ()
        return validate_loop_policies(
            await continuation_loop_policy_provider(session_id),
            field_name="continuation_loop_policies",
        )

    async def _resolve_session_query_authority_filters(
        *,
        parent_session_id: str | None,
        causal_budget_id: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve public list filters before they reach private store queries."""

        try:
            private_parent_session_id = _clean_optional_query_value(
                parent_session_id,
                "parent_session_id",
            )
            private_causal_budget_id = _clean_optional_query_value(
                causal_budget_id,
                "causal_budget_id",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if private_parent_session_id is not None:
            try:
                private_parent_session_id = await cayu_app._resolve_public_parent_session_id(
                    private_parent_session_id
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        if private_causal_budget_id is not None:
            try:
                private_causal_budget_id = await cayu_app._resolve_public_causal_budget_id(
                    private_causal_budget_id
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return private_parent_session_id, private_causal_budget_id

    async def _resolve_public_interaction_id(
        *,
        session_id: str,
        value: str,
    ) -> str:
        try:
            return await cayu_app._resolve_public_interaction_id(
                session_id=session_id,
                value=value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def _resolve_public_linkage(
        *,
        session_id: str,
        value: str,
        field_name: str,
    ) -> str:
        """Resolve public and transitional raw linkage through the app boundary."""

        try:
            return await cayu_app._resolve_public_action_linkage(
                session_id=session_id,
                value=value,
                field_name=field_name,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    async def _private_accepted_event(
        accepted_event: Event,
        *,
        session_id: str,
        publication_uncertain: bool,
    ) -> tuple[Event, int | None]:
        """Resolve one streamed or uncertain terminal event to durable authority."""

        public_session_id = cayu_app.project_session_id_for_exposure(session_id)
        if accepted_event.session_id not in {session_id, public_session_id}:
            raise RuntimeError(
                "Mutation acceptance event belongs to a different session: "
                f"{accepted_event.session_id}"
            )
        sequence = event_durable_sequence(accepted_event)
        try:
            if sequence is not None:
                records = await session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        after_sequence=sequence - 1,
                        limit=1,
                    )
                )
                if len(records) != 1 or records[0].sequence != sequence:
                    records = []
            else:
                records = await session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_id=accepted_event.id,
                        limit=1,
                    )
                )
        except Exception:
            if not publication_uncertain:
                raise
            records = []
        if records:
            private_event = records[0].event
            if private_event.session_id != session_id:
                raise RuntimeError(
                    "Mutation acceptance event belongs to a different session: "
                    f"{private_event.session_id}"
                )
            return private_event, records[0].sequence
        if not publication_uncertain or not event_id_is_runtime_generated(accepted_event):
            raise RuntimeError("Mutation acceptance boundary has no unique durable sequence.")
        # The terminal append may not have committed, so no public sequence alias
        # exists. Preserve its positively runtime-generated identity only inside
        # the private marker; public projection replaces it with a fixed sentinel.
        return accepted_event, None

    def _mutation_acceptance_callbacks(
        *,
        mutation_id: str | None,
        mutation_kind: str,
        session_id: str,
        after_accept: Callable[[Event], Awaitable[None]] | None = None,
    ) -> _MutationAcceptanceCallbacks:
        """Compose route bookkeeping with an exact durable mutation boundary."""
        if mutation_id is not None:
            projected_mutation_id = cayu_app.redact_json(mutation_id)
            if type(projected_mutation_id) is not str:
                raise RuntimeError("Mutation-id projection returned a non-string value.")
            if projected_mutation_id != mutation_id:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Cayu-Mutation-ID contains a configured workload secret and "
                        "cannot be used as durable mutation authority."
                    ),
                )

        async def record_acceptance(
            accepted_event: Event,
            *,
            publication_uncertain: bool,
        ) -> None:
            # Route-owned setup remains part of acceptance. The mutation marker
            # is deliberately written last so it never claims that a request was
            # accepted when prerequisite setup failed.
            private_accepted_event, accepted_event_sequence = await _private_accepted_event(
                accepted_event,
                session_id=session_id,
                publication_uncertain=publication_uncertain,
            )
            if after_accept is not None:
                await after_accept(private_accepted_event)
            if mutation_id is None:
                return
            payload: dict[str, Any] = {
                "mutation_id": mutation_id,
                "mutation_kind": mutation_kind,
                "accepted_event_type": str(private_accepted_event.type),
            }
            if accepted_event_sequence is None:
                payload["accepted_event_id"] = private_accepted_event.id
            else:
                payload["accepted_event_sequence"] = accepted_event_sequence
            if publication_uncertain:
                # The terminal status is durable, but its preassigned event may
                # have committed before this marker or may be repaired after it.
                # Clients use this explicit exception to accept either ordering
                # without weakening normal mutation-boundary validation.
                payload["accepted_event_publication_uncertain"] = True
            marker = Event(
                type=EventType.SERVER_MUTATION_ACCEPTED,
                session_id=session_id,
                interaction_id=private_accepted_event.interaction_id,
                agent_name=private_accepted_event.agent_name,
                environment_name=private_accepted_event.environment_name,
                workflow_name=private_accepted_event.workflow_name,
                tool_name=private_accepted_event.tool_name,
                payload=payload,
            )
            if accepted_event_sequence is None:
                marker = event_with_runtime_payload_authority(marker, "accepted_event_id")
            await cayu_app._emit_event_private(marker)

        async def after_first_event(first_event: Event) -> None:
            await record_acceptance(first_event, publication_uncertain=False)

        async def after_terminal_publication_uncertain(
            error: TerminalEventPublicationUncertain,
        ) -> None:
            await record_acceptance(error.event, publication_uncertain=True)

        if mutation_id is None:
            return _MutationAcceptanceCallbacks(
                after_first_event=after_first_event if after_accept is not None else None,
                after_terminal_publication_uncertain=None,
            )
        return _MutationAcceptanceCallbacks(
            after_first_event=after_first_event,
            after_terminal_publication_uncertain=after_terminal_publication_uncertain,
        )

    def _promotion_error_detail(
        code: str,
        message: str,
        *,
        reason: str | None = None,
    ) -> dict[str, str]:
        detail = {"code": code, "message": message}
        if reason is not None:
            detail["reason"] = reason
        return detail

    def _raise_promotion_trajectory_error(exc: SessionTrajectoryError) -> NoReturn:
        terminal_code = exc.terminal_code
        reason = terminal_code.value if terminal_code is not None else exc.code.value
        if terminal_code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND:
            status_code = 404
            code = "session_not_found"
        elif terminal_code in {
            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
            TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
        } or exc.code in {
            SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED,
            SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED,
        }:
            status_code = 413
            code = "evidence_limit_exceeded"
        else:
            status_code = 409
            code = "source_ineligible"
        raise HTTPException(
            status_code=status_code,
            detail=_promotion_error_detail(code, str(exc), reason=reason),
        ) from exc

    async def _load_promotion_baseline(
        public_session_id: str,
    ) -> tuple[Trajectory, PromotionCandidateV1]:
        assert evaluation_promotion is not None
        private_session_id = await _resolve_public_session_id(public_session_id)
        try:
            trajectory = await trajectory_from_session(cayu_app, private_session_id)
        except SessionTrajectoryError as exc:
            _raise_promotion_trajectory_error(exc)
        try:
            candidate = build_promotion_candidate(
                cayu_app,
                trajectory,
                target_key=evaluation_promotion.target_key,
                source_agent_name=evaluation_promotion.source_agent_name,
                application_release_id=evaluation_promotion.application_release_id,
                evidence_policy=evaluation_promotion.evidence_policy,
                pricing=evaluation_promotion_pricing,
            )
        except SessionPromotionError as exc:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "source_ineligible",
                    str(exc),
                    reason=exc.code.value,
                ),
            ) from exc
        return trajectory, candidate

    def _promotion_candidate_from_draft(
        baseline: PromotionCandidateV1,
        draft: EvaluationPromotionDraft,
    ) -> PromotionCandidateV1:
        if draft.expected_baseline_revision != baseline.revision:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The promotion baseline changed; preview the session again.",
                ),
            )
        _require_safe_promotion_document(
            draft.model_dump(mode="json"),
            code="draft_rejected",
            failure_subject="The edited candidate",
        )
        try:
            suite = EvalSuiteSpec.create(
                id=draft.suite.id,
                name=draft.suite.name,
                description=draft.suite.description,
                trial_request=draft.suite.trial_request,
            )
            case = EvalCaseSpec.create(
                id=draft.case.id,
                suite_id=draft.case.suite_id,
                name=draft.case.name,
                description=draft.case.description,
                source=baseline.source.case_source(),
                input=draft.case.input,
                assertions=draft.case.assertions,
            )
            candidate = PromotionCandidateV1.create(
                target_key=baseline.target_key,
                source=baseline.source,
                evidence_policy=baseline.evidence_policy,
                pricing_profile=baseline.pricing_profile,
                evidence=baseline.evidence,
                suite=suite,
                case=case,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "draft_rejected",
                    "The edited candidate violates the promotion contract.",
                ),
            ) from exc
        try:
            # A successful dashboard preview is also the export gate. Validate
            # corpus-only invariants here so the UI never presents a current
            # preview that the unchanged export route must reject later.
            corpus_from_promotion_candidate(candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "draft_rejected",
                    "The edited candidate is incompatible with the configured corpus limits or "
                    "pricing profile.",
                ),
            ) from exc
        return candidate

    def _require_safe_promotion_document(
        document: dict[str, Any],
        *,
        code: str,
        failure_subject: str,
    ) -> None:
        try:
            redacted_document = cayu_app.redact_json(document)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    code,
                    f"{failure_subject} could not cross the application redaction boundary.",
                ),
            ) from exc
        if redacted_document != document:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    code,
                    f"{failure_subject} contains a workload secret.",
                ),
            )

    def _promotion_server_fields_match(
        candidate: PromotionCandidateV1,
        baseline: PromotionCandidateV1,
    ) -> bool:
        return (
            candidate.target_key == baseline.target_key
            and candidate.source == baseline.source
            and candidate.evidence_policy == baseline.evidence_policy
            and candidate.pricing_profile == baseline.pricing_profile
            and candidate.evidence == baseline.evidence
            and candidate.warnings == baseline.warnings
        )

    @router.get(
        "/contract",
        response_model=ServerContractResponse,
        description=(
            "Return the versioned Cayu server contract. Authentication controls access "
            "to protected routes, but AuthContext.tenant is actor provenance only and "
            "does not filter or isolate Cayu data. Capability metadata supports "
            "presentation and discovery only; route enforcement remains authoritative."
        ),
        openapi_extra={"x-cayu-auth-context": auth_context_openapi_schema},
    )
    async def get_contract(
        response: Response,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        response.headers["Cache-Control"] = "private, no-store"
        return ServerContractResponse(
            api_prefix=api_prefix,
            client_generation=ClientGenerationContract(openapi_url=openapi_url),
            capabilities=capability_snapshot.project(
                auth_context,
                artifacts_configured=cayu_app.has_registered_artifact_store(),
            ),
        )

    if evaluation_promotion is not None:

        @bounded_evaluation_promotion_router.post(
            "/evals/promotion/sessions/{session_id}/preview",
            response_model=EvaluationPromotionPreviewResponse,
            responses=EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def preview_evaluation_promotion(
            session_id: str,
            body: EvaluationPromotionPreviewRequest,
        ) -> EvaluationPromotionPreviewResponse:
            trajectory, baseline = await _load_promotion_baseline(session_id)
            candidate = (
                baseline
                if body.draft is None
                else _promotion_candidate_from_draft(baseline, body.draft)
            )
            try:
                captured_score = score_promotion_candidate(
                    cayu_app,
                    trajectory,
                    candidate,
                    target_key=evaluation_promotion.target_key,
                    source_agent_name=evaluation_promotion.source_agent_name,
                    application_release_id=evaluation_promotion.application_release_id,
                    pricing=evaluation_promotion_pricing,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The captured evidence changed; preview the session again.",
                    ),
                ) from exc
            return EvaluationPromotionPreviewResponse(
                baseline_revision=baseline.revision,
                candidate=candidate,
                captured_score=captured_score,
            )

        @bounded_evaluation_promotion_router.post(
            "/evals/promotion/sessions/{session_id}/export",
            responses=EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
            dependencies=protected,
            response_class=Response,
        )
        async def export_evaluation_promotion(
            session_id: str,
            body: EvaluationPromotionExportRequest,
        ) -> Response:
            if body.expected_candidate_revision != body.candidate.revision:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The candidate changed after preview; preview it again before export.",
                    ),
                )
            _require_safe_promotion_document(
                body.candidate.model_dump(mode="json"),
                code="candidate_rejected",
                failure_subject="The candidate",
            )
            trajectory, baseline = await _load_promotion_baseline(session_id)
            if not _promotion_server_fields_match(body.candidate, baseline):
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The captured evidence or configured promotion identity changed.",
                    ),
                )
            try:
                score_promotion_candidate(
                    cayu_app,
                    trajectory,
                    body.candidate,
                    target_key=evaluation_promotion.target_key,
                    source_agent_name=evaluation_promotion.source_agent_name,
                    application_release_id=evaluation_promotion.application_release_id,
                    pricing=evaluation_promotion_pricing,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The candidate is no longer exportable; preview it again.",
                    ),
                ) from exc
            try:
                corpus_bytes = export_promotion_corpus(body.candidate)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=_promotion_error_detail(
                        "candidate_rejected",
                        "The candidate cannot be exported as a portable corpus.",
                    ),
                ) from exc
            return Response(
                content=corpus_bytes,
                media_type="application/json",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{evaluation_promotion.target_key}.eval.json"'
                    )
                },
            )

    if eval_registry is not None:

        @bounded_evals_router.get(
            "/evals/targets",
            response_model=EvalTargetCatalogResponse,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_targets() -> EvalTargetCatalogResponse:
            return eval_registry.catalog()

        captured_eval_store = None if eval_runtime is None else eval_runtime.store

        async def _load_captured_evaluation_baseline(
            public_session_id: str,
        ) -> tuple[Trajectory, EvalTargetRegistration, CapturedEvaluationCandidateV1]:
            private_session_id = await _resolve_public_session_id(public_session_id)
            try:
                trajectory = await trajectory_from_session(cayu_app, private_session_id)
            except SessionTrajectoryError as exc:
                _raise_promotion_trajectory_error(exc)
            session = trajectory.session
            if session is None:
                raise HTTPException(status_code=409, detail="Captured session evidence is absent.")
            registration = eval_registry.registration_for_agent(session.agent_name)
            if registration is None:
                raise HTTPException(
                    status_code=409,
                    detail="The session agent has no unambiguous published eval target.",
                )
            target = registration.target
            try:
                candidate = build_captured_evaluation_candidate(
                    cayu_app,
                    trajectory,
                    target_key=target.key,
                    source_agent_name=target.request_base.agent_name,
                    application_release_id=target.application_release_id,
                    evidence_policy=target.evidence_policy,
                    pricing=target.price_book,
                    project_root=registration.manifest_project_root,
                )
            except SessionPromotionError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "source_ineligible",
                        str(exc),
                        reason=exc.code.value,
                    ),
                ) from exc
            return trajectory, registration, candidate

        def _captured_candidate_from_draft(
            baseline: CapturedEvaluationCandidateV1,
            draft: CapturedEvaluationDraft,
        ) -> CapturedEvaluationCandidateV1:
            if draft.expected_baseline_revision != baseline.revision:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The captured evidence changed; preview the session again.",
                    ),
                )
            _require_safe_promotion_document(
                draft.model_dump(mode="json"),
                code="draft_rejected",
                failure_subject="The edited captured evaluation",
            )
            try:
                suite = EvalSuiteSpec.create(
                    id=draft.suite.id,
                    name=draft.suite.name,
                    description=draft.suite.description,
                )
                case = EvalCaseSpec.create(
                    id=draft.case.id,
                    suite_id=draft.case.suite_id,
                    name=draft.case.name,
                    description=draft.case.description,
                    source=baseline.source.case_source(),
                    input=None,
                    assertions=draft.case.assertions,
                )
                candidate = CapturedEvaluationCandidateV1.create(
                    target_key=baseline.target_key,
                    source=baseline.source,
                    evidence_policy=baseline.evidence_policy,
                    pricing_profile=baseline.pricing_profile,
                    evidence=baseline.evidence,
                    suite=suite,
                    case=case,
                )
                corpus_from_captured_evaluation_candidate(candidate)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=_promotion_error_detail(
                        "draft_rejected",
                        "The edited captured evaluation violates its portable contract.",
                    ),
                ) from exc
            return candidate

        def _captured_server_fields_match(
            candidate: CapturedEvaluationCandidateV1,
            baseline: CapturedEvaluationCandidateV1,
        ) -> bool:
            return (
                candidate.target_key == baseline.target_key
                and candidate.source == baseline.source
                and candidate.evidence_policy == baseline.evidence_policy
                and candidate.pricing_profile == baseline.pricing_profile
                and candidate.evidence == baseline.evidence
                and candidate.warnings == baseline.warnings
            )

        def _runnable_conversion(
            trajectory: Trajectory,
            registration: EvalTargetRegistration,
        ) -> CapturedEvaluationConversion:
            target = registration.target
            try:
                build_promotion_candidate(
                    cayu_app,
                    trajectory,
                    target_key=target.key,
                    source_agent_name=target.request_base.agent_name,
                    application_release_id=target.application_release_id,
                    evidence_policy=target.evidence_policy,
                    pricing=target.price_book,
                    project_root=registration.manifest_project_root,
                )
            except SessionPromotionError as exc:
                return CapturedEvaluationConversion(
                    available=False,
                    reason_code=exc.code.value,
                )
            except (TypeError, ValueError):
                return CapturedEvaluationConversion(
                    available=False,
                    reason_code="conversion_contract_unavailable",
                )
            return CapturedEvaluationConversion(available=True)

        async def _require_current_captured_candidate(
            session_id: str,
            candidate: CapturedEvaluationCandidateV1,
            expected_revision: str,
        ) -> tuple[Trajectory, EvalTargetRegistration, CapturedEvaluationCandidateV1]:
            if expected_revision != candidate.revision:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The evaluation changed after preview; preview it again.",
                    ),
                )
            _require_safe_promotion_document(
                candidate.model_dump(mode="json"),
                code="candidate_rejected",
                failure_subject="The captured evaluation",
            )
            trajectory, registration, baseline = await _load_captured_evaluation_baseline(
                session_id
            )
            if not _captured_server_fields_match(candidate, baseline):
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "preview_stale",
                        "The captured evidence or target identity changed.",
                    ),
                )
            return trajectory, registration, baseline

        def _score_current_captured_candidate(
            trajectory: Trajectory,
            registration: EvalTargetRegistration,
            candidate: CapturedEvaluationCandidateV1,
        ) -> CapturedRunScoreV1:
            """Revalidate one current candidate through the side-effect-free scorer."""

            target = registration.target
            try:
                return score_captured_evaluation_candidate(
                    cayu_app,
                    trajectory,
                    candidate,
                    target_key=target.key,
                    source_agent_name=target.request_base.agent_name,
                    application_release_id=target.application_release_id,
                    pricing=target.price_book,
                    project_root=registration.manifest_project_root,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=_promotion_error_detail(
                        "candidate_rejected",
                        "The captured evaluation cannot be scored from its retained evidence.",
                    ),
                ) from exc

        @bounded_captured_evaluation_router.post(
            "/evals/sessions/{session_id}/evaluation/save",
            response_model=CapturedEvaluationSaveResponse,
            status_code=201,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def save_captured_evaluation(
            session_id: str,
            body: CapturedEvaluationSaveRequest,
        ) -> CapturedEvaluationSaveResponse:
            if captured_eval_store is None or not captured_eval_store.captured_results:
                raise HTTPException(
                    status_code=409,
                    detail="Durable captured-result persistence is not available.",
                )
            trajectory, registration, _ = await _require_current_captured_candidate(
                session_id,
                body.candidate,
                body.expected_candidate_revision,
            )
            try:
                score = _score_current_captured_candidate(
                    trajectory,
                    registration,
                    body.candidate,
                )
                corpus = corpus_from_captured_evaluation_candidate(body.candidate)
                result = CapturedEvaluationResultV1.create(
                    corpus=corpus,
                    target=EvalResultTargetIdentityV1(
                        target_key=body.candidate.target_key,
                        application_release_id=body.candidate.source.application_release_id,
                        app_manifest_schema_version=(
                            body.candidate.source.app_manifest_schema_version
                        ),
                        app_manifest_fingerprint=(body.candidate.source.app_manifest_fingerprint),
                    ),
                    score=score,
                )
                record = await captured_eval_store.save_captured_result(
                    corpus,
                    result,
                    redact_json=cayu_app.redact_json,
                )
            except (EvalCorpusConflict, EvalResultConflict) as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The immutable captured evaluation conflicts with stored content.",
                ) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The captured evaluation contains unsafe public data.",
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="The captured evaluation exceeds the server byte limit.",
                ) from exc
            return CapturedEvaluationSaveResponse(record=record, result=result)

        @bounded_captured_evaluation_router.post(
            "/evals/sessions/{session_id}/evaluation/preview",
            response_model=CapturedEvaluationPreviewResponse,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def preview_captured_evaluation(
            session_id: str,
            body: CapturedEvaluationPreviewRequest,
        ) -> CapturedEvaluationPreviewResponse:
            trajectory, registration, baseline = await _load_captured_evaluation_baseline(
                session_id
            )
            candidate = (
                baseline
                if body.draft is None
                else _captured_candidate_from_draft(baseline, body.draft)
            )
            captured_score = _score_current_captured_candidate(
                trajectory,
                registration,
                candidate,
            )
            return CapturedEvaluationPreviewResponse(
                baseline_revision=baseline.revision,
                candidate=candidate,
                captured_score=captured_score,
                runnable_conversion=_runnable_conversion(trajectory, registration),
            )

        @bounded_captured_evaluation_router.post(
            "/evals/sessions/{session_id}/evaluation/export",
            response_class=Response,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def export_captured_evaluation(
            session_id: str,
            body: CapturedEvaluationExportRequest,
        ) -> Response:
            trajectory, registration, _ = await _require_current_captured_candidate(
                session_id,
                body.candidate,
                body.expected_candidate_revision,
            )
            _score_current_captured_candidate(
                trajectory,
                registration,
                body.candidate,
            )
            content = export_captured_evaluation_corpus(body.candidate)
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{body.candidate.target_key}-captured.eval.json"'
                    )
                },
            )

        @bounded_captured_evaluation_router.get(
            "/evals/results",
            response_model=EvalResultPage,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_results(
            target_key: Annotated[str, Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS)],
            cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
            limit: Annotated[int, Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE)] = (
                EVAL_STORE_DEFAULT_PAGE_SIZE
            ),
            max_result_bytes: Annotated[
                int,
                Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
            ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
            origin: Annotated[EvalResultOrigin | None, Query()] = None,
        ) -> EvalResultPage:
            if captured_eval_store is None or not captured_eval_store.captured_results:
                raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
            if eval_registry.get(target_key) is None:
                raise HTTPException(status_code=404, detail="Eval target not found.")
            try:
                return await captured_eval_store.list_results(
                    EvalResultQuery(
                        target_key=target_key,
                        origin=origin,
                        cursor=cursor,
                        limit=limit,
                        max_result_bytes=max_result_bytes,
                    )
                )
            except NotImplementedError as exc:
                raise HTTPException(
                    status_code=409, detail="Eval result catalog is unavailable."
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval result catalog exceeds the requested byte limit.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

        @bounded_captured_evaluation_router.get(
            "/evals/results/{result_revision}",
            response_model=EvalResultDetailResponse,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def get_eval_result(result_revision: str) -> EvalResultDetailResponse:
            if captured_eval_store is None or not captured_eval_store.captured_results:
                raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
            try:
                record = await captured_eval_store.load_result_record(result_revision)
                result = await captured_eval_store.load_result_by_revision(result_revision)
            except NotImplementedError as exc:
                raise HTTPException(
                    status_code=409, detail="Eval result catalog is unavailable."
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(status_code=413, detail="Eval result is too large.") from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail="Invalid eval result revision."
                ) from exc
            if (
                record is None
                or result is None
                or eval_registry.get(record.target.target_key) is None
            ):
                raise HTTPException(status_code=404, detail="Eval result not found.")
            key = EvalBaselineKey(
                target_key=record.target.target_key,
                corpus_revision=record.corpus_revision,
                suite_id=record.suite_id,
            )
            baseline = await captured_eval_store.load_baseline(key)
            return EvalResultDetailResponse(record=record, result=result, baseline=baseline)

        @bounded_captured_evaluation_router.post(
            "/evals/results/{result_revision}/baseline",
            response_model=EvalBaselineSelectionResponse,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        )
        async def select_eval_baseline(
            result_revision: str,
            body: EvalBaselineSelectionRequest,
            auth_context: AuthContext | None = optional_auth_context,
        ) -> EvalBaselineSelectionResponse:
            if captured_eval_store is None or not captured_eval_store.captured_results:
                raise HTTPException(status_code=409, detail="Eval baselines are unavailable.")
            if body.result_revision != result_revision:
                raise HTTPException(
                    status_code=400,
                    detail="Result revision path and request body do not match.",
                )
            try:
                record = await captured_eval_store.load_result_record(result_revision)
            except (NotImplementedError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail="Invalid eval result revision."
                ) from exc
            if record is None or eval_registry.get(record.target.target_key) is None:
                raise HTTPException(status_code=404, detail="Eval result not found.")
            actor_id = (
                "cayu:trusted-local-development" if auth_context is None else auth_context.subject
            )
            key = EvalBaselineKey(
                target_key=record.target.target_key,
                corpus_revision=record.corpus_revision,
                suite_id=record.suite_id,
            )
            try:
                mutation = await captured_eval_store.set_baseline(
                    EvalBaselineUpdate(
                        key=key,
                        result_revision=result_revision,
                        expected_generation=body.expected_generation,
                        operation_id=body.operation_id,
                        actor_id=actor_id,
                    ),
                    redact_json=cayu_app.redact_json,
                )
            except EvalBaselineConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The authenticated baseline actor cannot cross the public boundary.",
                ) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid baseline selection.") from exc
            baseline = await captured_eval_store.load_baseline(key)
            if baseline is None:
                raise RuntimeError("Committed eval baseline is unavailable.")
            return EvalBaselineSelectionResponse(baseline=baseline, mutation=mutation)

    if eval_runtime is not None:
        eval_store = eval_runtime.store
        active_eval_registry = eval_runtime.registry

        def _eval_target(target_key: str | None = None):
            selected_key = (
                active_eval_registry.default_target_key if target_key is None else target_key
            )
            target = active_eval_registry.get(selected_key)
            if target is None:
                raise HTTPException(status_code=404, detail="Eval target not found.")
            return target

        def _eval_query_error() -> NoReturn:
            raise HTTPException(status_code=422, detail="Invalid Evals query.")

        def _eval_run_invocation(
            auth_context: AuthContext | None,
            *,
            max_steps: int | None,
            limits: RunLimits | None,
            cost_budget: EvalRunCostBudget | None,
        ) -> EvalRunInvocation:
            try:
                origin = (
                    None
                    if auth_context is None
                    else InvocationOrigin(
                        trust=InvocationOriginTrust.SERVER_VERIFIED,
                        subject=auth_context.subject,
                        tenant=auth_context.tenant,
                    )
                )
                return EvalRunInvocation(
                    source=SessionExecutionSource.HTTP_RUN,
                    origin=origin,
                    max_steps=max_steps,
                    limits=limits,
                    cost_budget=cost_budget,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Eval execution bounds or authenticated provenance are invalid.",
                ) from exc

        def _eval_idempotency_digest(target_key: str, idempotency_key: str) -> str:
            try:
                clean_key = require_clean_nonblank(idempotency_key, "Idempotency-Key")
                require_unicode_scalar_text(clean_key, "Idempotency-Key")
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid Idempotency-Key.") from exc
            return (
                "sha256:"
                + hashlib.sha256(
                    b"cayu-server-eval-idempotency-v1\0"
                    + target_key.encode("ascii")
                    + b"\0"
                    + clean_key.encode("utf-8")
                ).hexdigest()
            )

        async def _admit_eval_run(
            *,
            corpus: EvalCorpusDocument,
            max_concurrency: int,
            invocation: EvalRunInvocation,
            idempotency_key: str,
            eval_target: CorpusTarget,
            compiled: CompiledCorpusSuite,
        ) -> EvalRunRecord:
            if eval_target.key != corpus.target_key:
                raise RuntimeError("Prepared eval target does not match its corpus.")
            run_request = EvalRunRequest(
                run_id=f"eval-{uuid4().hex}",
                corpus_revision=corpus.revision,
                target_key=eval_target.key,
                suite_id=compiled.run_contract.suite_id,
                suite_revision=compiled.run_contract.suite_revision,
                max_concurrency=max_concurrency,
                invocation=invocation,
                idempotency_key=_eval_idempotency_digest(
                    eval_target.key,
                    idempotency_key,
                ),
            )
            try:
                return await eval_store.admit_run(
                    run_request,
                    redact_json=eval_target.app.redact_json,
                )
            except EvalRunAdmissionConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency-Key is already bound to another eval run request.",
                ) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Eval run request contains unsafe public data.",
                ) from exc

        async def _prepare_eval_run(
            *,
            corpus: EvalCorpusDocument,
            suite_id: str,
            max_concurrency: int,
            invocation: EvalRunInvocation,
        ) -> tuple[CorpusTarget, CompiledCorpusSuite]:
            eval_target = _eval_target(corpus.target_key)
            if any(case.suite_id == suite_id and case.input is None for case in corpus.cases):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This captured evaluation has no runnable input. Author runnable "
                        "input or a scenario before launching fresh work."
                    ),
                )
            try:
                effective_target = target_for_eval_invocation(eval_target, invocation)
                compiled = await asyncio.to_thread(
                    compile_corpus_suite,
                    corpus,
                    effective_target,
                    suite_id,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Eval run is incompatible with the attached target or bounds.",
                ) from exc
            if max_concurrency > eval_target.limits.max_concurrency:
                raise HTTPException(
                    status_code=400,
                    detail="Eval run exceeds the attached target concurrency limit.",
                )
            return eval_target, compiled

        async def _load_eval_corpus(corpus_revision: str) -> EvalCorpusDocument:
            try:
                corpus = await eval_store.load_corpus(corpus_revision)
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval corpus exceeds the server byte limit.",
                ) from exc
            except (TypeError, ValueError):
                _eval_query_error()
            if corpus is None:
                raise HTTPException(status_code=404, detail="Eval corpus not found.")
            if active_eval_registry.get(corpus.target_key) is None:
                raise HTTPException(status_code=404, detail="Eval corpus not found.")
            return corpus

        async def _load_eval_run(run_id: str):
            try:
                run = await eval_store.load_run(run_id)
            except (TypeError, ValueError):
                _eval_query_error()
            if run is None:
                raise HTTPException(status_code=404, detail="Eval run not found.")
            if active_eval_registry.get(run.spec.target_key) is None:
                raise HTTPException(status_code=404, detail="Eval run not found.")
            return run

        async def _load_eval_result(run_id: str):
            # Authorize the run before hydrating its potentially large result.
            await _load_eval_run(run_id)
            try:
                result = await eval_store.load_result(run_id)
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval result exceeds the server byte limit.",
                ) from exc
            if result is None:
                raise HTTPException(
                    status_code=409,
                    detail="Eval run has no completed result.",
                )
            # Result publication and run terminalization are one store transaction.
            # Reload after the result becomes visible so a concurrent publication
            # cannot pair it with the active record observed above.
            run = await _load_eval_run(run_id)
            return run, result

        @bounded_captured_evaluation_router.post(
            "/evals/sessions/{session_id}/evaluation/launch",
            response_model=CapturedEvaluationLaunchResponse,
            status_code=202,
            responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def launch_captured_evaluation(
            session_id: str,
            body: CapturedEvaluationLaunchRequest,
            idempotency_key: Annotated[
                str,
                Header(alias="Idempotency-Key", min_length=1, max_length=512),
            ],
            auth_context: AuthContext | None = optional_auth_context,
        ) -> CapturedEvaluationLaunchResponse:
            if not eval_store.captured_results:
                raise HTTPException(
                    status_code=409,
                    detail="Durable captured-result persistence is not available.",
                )
            trajectory, registration, _ = await _require_current_captured_candidate(
                session_id,
                body.candidate,
                body.expected_candidate_revision,
            )
            target = registration.target
            try:
                runnable_baseline = build_promotion_candidate(
                    cayu_app,
                    trajectory,
                    target_key=target.key,
                    source_agent_name=target.request_base.agent_name,
                    application_release_id=target.application_release_id,
                    evidence_policy=target.evidence_policy,
                    pricing=target.price_book,
                    project_root=registration.manifest_project_root,
                )
            except SessionPromotionError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=_promotion_error_detail(
                        "source_ineligible",
                        str(exc),
                        reason=exc.code.value,
                    ),
                ) from exc
            try:
                runnable_candidate = runnable_promotion_candidate(
                    body.candidate,
                    runnable_baseline,
                    trial_request=body.trial_request,
                )
                score = score_promotion_candidate(
                    cayu_app,
                    trajectory,
                    runnable_candidate,
                    target_key=target.key,
                    source_agent_name=target.request_base.agent_name,
                    application_release_id=target.application_release_id,
                    pricing=target.price_book,
                    project_root=registration.manifest_project_root,
                )
                corpus = corpus_from_promotion_candidate(runnable_candidate)
                result = CapturedEvaluationResultV1.create(
                    corpus=corpus,
                    target=EvalResultTargetIdentityV1(
                        target_key=runnable_candidate.target_key,
                        application_release_id=(runnable_candidate.source.application_release_id),
                        app_manifest_schema_version=(
                            runnable_candidate.source.app_manifest_schema_version
                        ),
                        app_manifest_fingerprint=(
                            runnable_candidate.source.app_manifest_fingerprint
                        ),
                    ),
                    score=score,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=_promotion_error_detail(
                        "candidate_rejected",
                        "The reviewed evaluation cannot be converted to runnable work.",
                    ),
                ) from exc
            invocation = _eval_run_invocation(
                auth_context,
                max_steps=body.max_steps,
                limits=body.limits,
                cost_budget=body.cost_budget,
            )
            eval_target, compiled = await _prepare_eval_run(
                corpus=corpus,
                suite_id=runnable_candidate.suite.id,
                max_concurrency=body.max_concurrency,
                invocation=invocation,
            )
            try:
                record = await eval_store.save_captured_result(
                    corpus,
                    result,
                    redact_json=target.app.redact_json,
                )
            except (EvalCorpusConflict, EvalResultConflict) as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The immutable runnable evaluation conflicts with stored content.",
                ) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The runnable evaluation contains unsafe public data.",
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="The runnable evaluation exceeds the server byte limit.",
                ) from exc
            run = await _admit_eval_run(
                corpus=corpus,
                max_concurrency=body.max_concurrency,
                invocation=invocation,
                idempotency_key=idempotency_key,
                eval_target=eval_target,
                compiled=compiled,
            )
            return CapturedEvaluationLaunchResponse(
                captured=CapturedEvaluationSaveResponse(record=record, result=result),
                run=run,
            )

        @bounded_evals_router.post(
            "/evals/corpora",
            response_model=EvalCorpusCatalogEntry,
            status_code=201,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
            openapi_extra=_json_request_openapi("EvalCorpusDocument"),
        )
        async def import_eval_corpus(request: Request):
            corpus = await _validated_private_json_body(
                request,
                EvalCorpusDocument,
                invalid_detail="Invalid Evals request.",
            )
            eval_target = active_eval_registry.get(corpus.target_key)
            if eval_target is None:
                raise HTTPException(
                    status_code=400,
                    detail="Eval corpus is incompatible with the attached target.",
                )
            try:
                await asyncio.to_thread(
                    evaluation_target_identity,
                    eval_target,
                )
                await asyncio.to_thread(
                    _validate_corpus_target_compatibility,
                    corpus,
                    eval_target,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Eval corpus is incompatible with the attached target.",
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Attached eval target is unavailable.",
                ) from exc
            try:
                return await eval_store.save_corpus(
                    corpus,
                    redact_json=eval_target.app.redact_json,
                )
            except EvalCorpusConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Eval corpus revision conflicts with stored content.",
                ) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Eval corpus contains unsafe public data.",
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval corpus exceeds the server byte limit.",
                ) from exc

        @bounded_evals_router.get(
            "/evals/corpora",
            response_model=EvalCorpusCatalogPage,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_corpora(
            target_key: Annotated[
                str | None,
                Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
            ] = None,
            cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
            limit: Annotated[
                int,
                Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
            ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
            max_result_bytes: Annotated[
                int,
                Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
            ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
        ):
            eval_target = _eval_target(target_key)
            try:
                return await eval_store.list_corpora(
                    EvalCatalogQuery(
                        target_key=eval_target.key,
                        cursor=cursor,
                        limit=limit,
                        max_result_bytes=max_result_bytes,
                    )
                )
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval catalog page exceeds the requested byte limit.",
                ) from exc
            except (TypeError, ValueError):
                _eval_query_error()

        @bounded_evals_router.get(
            "/evals/corpora/{corpus_revision}",
            response_model=EvalCorpusDocument,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def get_eval_corpus(corpus_revision: str):
            corpus = await _load_eval_corpus(corpus_revision)
            return await _model_json_response(corpus, EvalCorpusDocument)

        @bounded_evals_router.get(
            "/evals/corpora/{corpus_revision}/download",
            response_class=Response,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def download_eval_corpus(corpus_revision: str) -> Response:
            corpus = await _load_eval_corpus(corpus_revision)
            corpus_json = await asyncio.to_thread(_render_utf8, eval_corpus_to_json, corpus)
            return Response(
                content=corpus_json,
                media_type="application/json",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{corpus.target_key}-{corpus.revision[7:19]}.eval.json"'
                    )
                },
            )

        @bounded_evals_router.get(
            "/evals/corpora/{corpus_revision}/suites",
            response_model=EvalSuiteCatalogPage,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_suites(
            corpus_revision: str,
            cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
            limit: Annotated[
                int,
                Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
            ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
            max_result_bytes: Annotated[
                int,
                Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
            ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
        ):
            await _load_eval_corpus(corpus_revision)
            try:
                page = await eval_store.list_suites(
                    EvalSuiteCatalogQuery(
                        corpus_revision=corpus_revision,
                        cursor=cursor,
                        limit=limit,
                        max_result_bytes=max_result_bytes,
                    )
                )
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval suite page exceeds the requested byte limit.",
                ) from exc
            except (TypeError, ValueError):
                _eval_query_error()
            return await _model_json_response(page, EvalSuiteCatalogPage)

        @bounded_evals_router.get(
            "/evals/corpora/{corpus_revision}/suites/{suite_id}/cases",
            response_model=EvalCaseCatalogPage,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_cases(
            corpus_revision: str,
            suite_id: str,
            cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
            limit: Annotated[
                int,
                Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
            ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
            max_result_bytes: Annotated[
                int,
                Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
            ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
        ):
            corpus = await _load_eval_corpus(corpus_revision)
            if all(suite.id != suite_id for suite in corpus.suites):
                raise HTTPException(status_code=404, detail="Eval suite not found.")
            try:
                page = await eval_store.list_cases(
                    EvalCaseCatalogQuery(
                        corpus_revision=corpus_revision,
                        suite_id=suite_id,
                        cursor=cursor,
                        limit=limit,
                        max_result_bytes=max_result_bytes,
                    )
                )
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval case page exceeds the requested byte limit.",
                ) from exc
            except (TypeError, ValueError):
                _eval_query_error()
            return await _model_json_response(page, EvalCaseCatalogPage)

        @bounded_evals_router.post(
            "/evals/runs",
            response_model=EvalRunRecord,
            status_code=202,
            responses=EVALS_ENDPOINT_RESPONSES,
            openapi_extra=_json_request_openapi(EvalRunCreateRequest),
        )
        async def create_eval_run(
            request: Request,
            idempotency_key: Annotated[
                str,
                Header(alias="Idempotency-Key", min_length=1, max_length=512),
            ],
            auth_context: AuthContext | None = optional_auth_context,
        ):
            body = await _validated_private_json_body(
                request,
                EvalRunCreateRequest,
                invalid_detail="Invalid Evals request.",
            )
            corpus = await _load_eval_corpus(body.corpus_revision)
            invocation = _eval_run_invocation(
                auth_context,
                max_steps=body.max_steps,
                limits=body.limits,
                cost_budget=body.cost_budget,
            )
            eval_target, compiled = await _prepare_eval_run(
                corpus=corpus,
                suite_id=body.suite_id,
                max_concurrency=body.max_concurrency,
                invocation=invocation,
            )
            return await _admit_eval_run(
                corpus=corpus,
                max_concurrency=body.max_concurrency,
                invocation=invocation,
                idempotency_key=idempotency_key,
                eval_target=eval_target,
                compiled=compiled,
            )

        @bounded_evals_router.get(
            "/evals/runs",
            response_model=EvalRunPage,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def list_eval_runs(
            target_key: Annotated[
                str | None,
                Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
            ] = None,
            status: EvalRunStatus | None = None,
            corpus_revision: str | None = None,
            cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
            limit: Annotated[
                int,
                Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
            ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
            max_result_bytes: Annotated[
                int,
                Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
            ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
        ):
            eval_target = _eval_target(target_key)
            try:
                return await eval_store.list_runs(
                    EvalRunQuery(
                        target_key=eval_target.key,
                        status=status,
                        corpus_revision=corpus_revision,
                        cursor=cursor,
                        limit=limit,
                        max_result_bytes=max_result_bytes,
                    )
                )
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval run page exceeds the requested byte limit.",
                ) from exc
            except (TypeError, ValueError):
                _eval_query_error()

        @bounded_evals_router.get(
            "/evals/runs/{run_id}",
            response_model=EvalRunRecord,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def get_eval_run(run_id: str):
            return await _load_eval_run(run_id)

        @bounded_evals_router.post(
            "/evals/runs/{run_id}/cancel",
            response_model=EvalRunRecord,
            status_code=202,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def cancel_eval_run(run_id: str):
            await _load_eval_run(run_id)
            try:
                return await eval_store.request_cancel(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Eval run not found.") from exc
            except (TypeError, ValueError):
                _eval_query_error()

        @bounded_evals_router.get(
            "/evals/runs/{run_id}/result",
            response_model=EvalResultResponse,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def get_eval_result(run_id: str) -> Response:
            run, result = await _load_eval_result(run_id)
            response = await asyncio.to_thread(EvalResultResponse, run=run, result=result)
            return await _model_json_response(response, EvalResultResponse)

        @bounded_evals_router.get(
            "/evals/runs/{run_id}/report.json",
            response_class=Response,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def download_eval_json_report(run_id: str) -> Response:
            _, result = await _load_eval_result(run_id)
            report = await asyncio.to_thread(
                _render_utf8,
                corpus_execution_result_to_json,
                result,
            )
            return Response(
                content=report,
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="{run_id}.eval-result.json"'
                },
            )

        @bounded_evals_router.get(
            "/evals/runs/{run_id}/report.html",
            response_class=Response,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
        )
        async def download_eval_html_report(run_id: str) -> Response:
            _, result = await _load_eval_result(run_id)
            report = await asyncio.to_thread(
                _render_utf8,
                render_corpus_execution_html,
                result,
            )
            return Response(
                content=report,
                media_type="text/html",
                headers={
                    "Content-Disposition": f'attachment; filename="{run_id}.eval-report.html"'
                },
            )

        @bounded_evals_router.post(
            "/evals/comparisons",
            response_model=EvalComparisonResponse,
            responses=EVALS_ENDPOINT_RESPONSES,
            dependencies=protected,
            openapi_extra=_json_request_openapi(EvalComparisonRequest),
        )
        async def compare_eval_runs(request: Request) -> Response:
            body = await _validated_private_json_body(
                request,
                EvalComparisonRequest,
                invalid_detail="Invalid Evals request.",
            )
            baseline_run, baseline = await _load_eval_result(body.baseline_run_id)
            if body.current_run_id == body.baseline_run_id:
                current_run, current = baseline_run, baseline
            else:
                current_run, current = await _load_eval_result(body.current_run_id)
            comparison = await asyncio.to_thread(
                compare_corpus_execution_results,
                baseline,
                current,
            )
            response = EvalComparisonResponse(
                baseline=baseline_run,
                current=current_run,
                comparison=comparison,
            )
            return await _model_json_response(response, EvalComparisonResponse)

    @router.get(
        "/system/diagnostics",
        response_model=SystemDiagnosticsResponse,
        description=(
            "Return bounded protected Cayu configuration and registration diagnostics. "
            "This is an explicit operator snapshot, not readiness, infrastructure "
            "monitoring, or an authorization token. It performs no dependency probes."
        ),
        openapi_extra={"x-cayu-auth-context": auth_context_openapi_schema},
    )
    async def get_system_diagnostics(
        response: Response,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        response.headers["Cache-Control"] = "private, no-store"
        fingerprints, total_count = cayu_app.artifact_store_registration_fingerprints(
            limit=MAX_SYSTEM_ARTIFACT_STORE_REGISTRATIONS,
        )
        return diagnostics_snapshot.project(
            auth_context,
            artifact_store_fingerprints=fingerprints,
            artifact_store_total_count=total_count,
        )

    @router.post(
        "/operations/snapshot",
        response_model=OperationalSnapshotResponse,
        responses=AGGREGATE_ENDPOINT_RESPONSES,
        dependencies=protected,
        description=(
            "Return exact current status counts from each configured store. Session and "
            "task sections are separate store-local snapshots and are not presented as "
            "one cross-store atomic read. Scope is the configured stores; authentication "
            "does not add tenant filtering."
        ),
    )
    async def get_operational_snapshot(body: OperationalSnapshotRequest):
        try:
            session_snapshot = await session_store.aggregate_operational_snapshot(
                body.session_filter
            )
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail="The configured session store does not support aggregate snapshots.",
            ) from exc

        task_snapshot = None
        if not body.include_tasks:
            task_snapshot_status = "not_requested"
        elif task_store is None:
            task_snapshot_status = "not_configured"
        else:
            try:
                task_snapshot = await task_store.aggregate_operational_snapshot(body.task_filter)
            except NotImplementedError:
                task_snapshot_status = "unsupported"
            else:
                task_snapshot_status = "available"
        return OperationalSnapshotResponse(
            scope="configured_stores",
            cross_store_atomic=False,
            sessions=session_snapshot,
            task_snapshot_status=task_snapshot_status,
            tasks=task_snapshot,
        )

    async def get_usage_rollup(body: UsageRollupRequest, response: Response):
        response.headers["Cache-Control"] = "private, no-store"
        query = UsageRollupQuery(
            start_at=body.start_at,
            end_at=body.end_at,
            sessions=body.session_filter,
            group_limit=body.group_limit,
            session_group_limit=body.session_group_limit,
            include_pricing_inputs=body.pricing is not None,
            pricing_input_limit=body.pricing_input_limit,
        )
        try:
            result = await session_store.aggregate_usage(query)
            if type(result) is not UsageRollupStoreResult:
                raise UsageRollupInconsistent(
                    "Usage aggregate stores must return UsageRollupStoreResult."
                )
            result = result.validate_for_query(query)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail="The configured session store does not support usage aggregates.",
            ) from exc
        except UsageRollupResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Usage rollup result exceeds the server byte limit.",
                headers={"Cache-Control": "private, no-store"},
            ) from exc
        except (UsageRollupInconsistent, ValidationError) as exc:
            raise HTTPException(
                status_code=500,
                detail="The configured session store returned an inconsistent usage projection.",
                headers={"Cache-Control": "private, no-store"},
            ) from exc
        _require_safe_usage_session_authority(cayu_app, result)
        cost = (
            None
            if body.pricing is None
            else estimate_usage_rollup_cost(
                result,
                body.pricing,
                billing_group_limit=body.group_limit,
            )
        )
        session_cost_breakdown = (
            None
            if body.pricing is None or result.session_breakdown is None
            else estimate_usage_session_cost_breakdown(result, body.pricing)
        )
        response_value = UsageRollupResponse(
            scope="configured_session_store",
            time_basis="event.timestamp",
            session_filter_basis="current_session_attributes",
            as_of=result.as_of,
            start_at=result.start_at,
            end_at=result.end_at,
            accuracy=result.totals_accuracy,
            matching_session_count=result.matching_session_count,
            active_session_count=result.active_session_count,
            includes_active_sessions=result.includes_active_sessions,
            totals=result.totals,
            provider_breakdown=result.provider_breakdown,
            model_breakdown=result.model_breakdown,
            cost=cost,
            session_breakdown=result.session_breakdown,
            session_cost_breakdown=session_cost_breakdown,
        )
        if not json_utf8_size_within_limit(
            response_value.model_dump(mode="json"),
            MAX_USAGE_ROLLUP_RESULT_BYTES,
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "Usage rollup exceeds the 4 MiB serialized response limit. "
                    "Reduce session_group_limit, group_limit, or pricing identity size."
                ),
                headers={"Cache-Control": "private, no-store"},
            )
        return response_value

    router.add_api_route(
        "/usage/rollup",
        get_usage_rollup,
        methods=["POST"],
        response_model=UsageRollupResponse,
        responses=USAGE_ROLLUP_ENDPOINT_RESPONSES,
        dependencies=protected,
        description=(
            "Aggregate usage-bearing events in a UTC-normalized half-open event-time "
            "window. Session filters apply to current session attributes. Totals remain "
            "exact when provider/model detail is bounded into an explicit remainder; "
            "callers may opt into a bounded per-session breakdown with its own exact "
            "remainder; "
            "cost is omitted from evaluation rather than reported partially when its "
            "price-input bound is exceeded. Scope is the configured session store; "
            "authentication does not add tenant filtering."
        ),
        route_class_override=_BoundedUsageRollupRoute,
    )

    async def _public_or_legacy_event_record(
        session_id: str,
        event_id: str,
    ) -> tuple[EventRecord | None, bool]:
        """Resolve a public alias without shadowing a legacy raw event ID."""

        if not event_id.startswith(PUBLIC_EVENT_ID_PREFIX):
            return None, False
        raw_records = await session_store.query_events(
            EventQuery(session_id=session_id, event_id=event_id, limit=1)
        )
        raw_record = raw_records[0] if raw_records else None
        sequence = public_event_sequence(event_id)
        if sequence is None:
            if raw_record is not None:
                return raw_record, True
            raise HTTPException(
                status_code=422,
                detail="Event reference contains a malformed Cayu public event alias.",
            )
        alias_records = await session_store.query_events(
            EventQuery(
                session_id=session_id,
                after_sequence=sequence - 1,
                limit=1,
            )
        )
        alias_record = (
            alias_records[0] if alias_records and alias_records[0].sequence == sequence else None
        )
        if (
            raw_record is not None
            and alias_record is not None
            and raw_record.sequence != alias_record.sequence
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Event reference is ambiguous between a legacy raw ID and "
                    "a Cayu public event alias."
                ),
            )
        return raw_record or alias_record, True

    async def _marker_record(session_id: str, event_id: str) -> EventRecord:
        """Persisted event named by a ``Last-Event-ID`` marker.

        Unknown event markers are rejected rather than silently widening the replay
        to full history. The explicit ``session_id:`` marker owns replay-from-start.
        """
        resolved, handled = await _public_or_legacy_event_record(session_id, event_id)
        if not handled:
            records = await session_store.query_events(
                EventQuery(session_id=session_id, event_id=event_id, limit=1)
            )
            resolved = records[0] if records else None
        if resolved is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Last-Event-ID event was not found in the requested session. "
                    f"Use `{SSE_REPLAY_START_MARKER_FORMAT}` only for explicit "
                    "replay from the beginning."
                ),
            )
        return resolved

    async def _marker_terminal_boundary_id(marker_record: EventRecord) -> str | None:
        """Resolve terminal lineage represented by a durable replay marker.

        Terminal events carry their own lineage. Terminal-hook telemetry names its
        terminal event explicitly, but that reference still has to resolve to a
        matching durable record in the same operation epoch. Interruption-cascade
        telemetry is also runtime-owned and post-terminal, but does not carry an
        explicit reference, so resolve its latest prior terminal event. Reject both
        indirect associations if a later operation started before the marker.
        """
        marker_event = marker_record.event
        if marker_event.type in _REPLAY_TERMINAL_EVENT_TYPES:
            return marker_event.id

        terminal_record: EventRecord | None = None
        if marker_event.type in _REPLAY_TERMINAL_LINEAGE_EVENT_TYPES:
            referenced_terminal_id = _replay_terminal_boundary_id(marker_event)
            if referenced_terminal_id is None:
                return None
            terminal_records = await session_store.query_events(
                EventQuery(
                    session_id=marker_event.session_id,
                    event_id=referenced_terminal_id,
                    limit=1,
                )
            )
            if not terminal_records:
                return None
            candidate = terminal_records[0]
            if (
                candidate.sequence >= marker_record.sequence
                or candidate.event.type != marker_event.payload.get("terminal_event_type")
                or candidate.event.type not in _REPLAY_TERMINAL_EVENT_TYPES
            ):
                return None
            terminal_record = candidate
        elif marker_event.type in _REPLAY_POST_TERMINAL_EVENT_TYPES:
            terminal_records = await session_store.query_events(
                EventQuery(
                    session_id=marker_event.session_id,
                    event_types=tuple(sorted(_REPLAY_TERMINAL_EVENT_TYPES, key=str)),
                    before_sequence=marker_record.sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if terminal_records:
                terminal_record = terminal_records[0]
        else:
            return None

        if terminal_record is None:
            return None
        later_operation_starts = await session_store.query_events(
            EventQuery(
                session_id=marker_event.session_id,
                event_types=tuple(sorted(_REPLAY_OPERATION_START_EVENT_TYPES, key=str)),
                after_sequence=terminal_record.sequence,
                before_sequence=marker_record.sequence,
                limit=1,
            )
        )
        if later_operation_starts:
            return None
        return terminal_record.event.id

    async def _replay_events_response(
        http_request: Request,
        *,
        expected_session_id: str | None = None,
    ) -> EventSourceResponse | None:
        """SSE resume for reconnecting clients (``Last-Event-ID`` header).

        Instead of starting new work, replay the session's persisted events after the
        last one the client saw and keep following until the session reaches a
        terminal status (the detached pump finishes the run even after a disconnect).
        """
        last_event_id = http_request.headers.get("last-event-id")
        if last_event_id is None:
            return None
        marker_record: EventRecord | None = None
        public_expected_session_id: str | None = None
        if expected_session_id is not None:
            expected_session_id = await _resolve_public_session_id(expected_session_id)
            public_expected_session_id = cayu_app.project_session_id_for_exposure(
                expected_session_id
            )
        marker = parse_last_event_id(
            last_event_id,
            expected_session_id=expected_session_id,
            public_session_id=public_expected_session_id,
        )
        if marker is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Last-Event-ID must use a session-bound Cayu public event id "
                    f"or the explicit `{SSE_REPLAY_START_MARKER_FORMAT}` start marker."
                ),
            )
        session_id, last_seen_event_id = marker
        if expected_session_id is not None:
            if session_id == public_expected_session_id:
                # A legacy/imported secret-bearing session has a safe marker on
                # the wire. The request's private session identity scopes this
                # otherwise non-unique presentation value before any query.
                session_id = expected_session_id
            elif session_id != expected_session_id:
                raise HTTPException(
                    status_code=422,
                    detail="Last-Event-ID session does not match the request session_id.",
                )
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Session not found: {cayu_app.project_session_id_for_exposure(session_id)}"
                ),
            )
        if last_seen_event_id is not None:
            # Public aliases and legacy raw IDs share one disambiguating lookup.
            marker_record = await _marker_record(session_id, last_seen_event_id)
        after_sequence = None if marker_record is None else marker_record.sequence

        async def replay() -> AsyncIterator[dict[str, str]]:
            replay_after_sequence = after_sequence
            operation_in_progress = state.status in _REPLAY_ACTIVE_SESSION_STATUSES
            observed_terminal_event_id = (
                None
                if marker_record is None or operation_in_progress
                else await _marker_terminal_boundary_id(marker_record)
            )
            loop = asyncio.get_running_loop()
            idle_deadline = loop.time() + replay_idle_timeout_s
            poll_interval = _REPLAY_POLL_INTERVAL_MIN_S

            while True:
                page = await session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        after_sequence=replay_after_sequence,
                        limit=SSE_REPLAY_PAGE_EVENTS,
                    )
                )
                for record in page:
                    try:
                        message = event_to_sse_message(
                            cayu_app._project_persisted_event_record_for_exposure(record).event
                        )
                    except SseEventFrameTooLargeError as exc:
                        yield _stream_error_sse_message(
                            cayu_app,
                            exc,
                            kind="observer",
                            code="event_frame_too_large",
                            retryable=False,
                            session_id=session_id,
                        )
                        return
                    replay_after_sequence = record.sequence
                    terminal_boundary_id = _replay_terminal_boundary_id(record.event)
                    if record.event.type in _REPLAY_TERMINAL_EVENT_TYPES:
                        observed_terminal_event_id = terminal_boundary_id
                        operation_in_progress = False
                    elif record.event.type in _REPLAY_OPERATION_START_EVENT_TYPES:
                        observed_terminal_event_id = None
                        operation_in_progress = True
                    elif (
                        terminal_boundary_id is not None
                        and not operation_in_progress
                        and terminal_boundary_id != observed_terminal_event_id
                    ):
                        # A hook may establish a boundary when replay began after
                        # the terminal event. Validate its durable reference and
                        # operation epoch before trusting payload-carried lineage.
                        validated_boundary_id = await _marker_terminal_boundary_id(record)
                        if validated_boundary_id is not None:
                            observed_terminal_event_id = validated_boundary_id
                    elif operation_in_progress:
                        observed_terminal_event_id = None
                    yield message
                if page:
                    idle_deadline = loop.time() + replay_idle_timeout_s
                    poll_interval = _REPLAY_POLL_INTERVAL_MIN_S
                if len(page) == SSE_REPLAY_PAGE_EVENTS:
                    continue
                current = await session_store.load_state(session_id)
                if current is None:
                    return
                terminal_event_type = _REPLAY_TERMINAL_EVENT_BY_STATUS.get(current.status)
                if terminal_event_type is not None:
                    terminal_records = await session_store.query_events(
                        EventQuery(
                            session_id=session_id,
                            event_type=terminal_event_type,
                            order_by=EventOrder.SEQUENCE_DESC,
                            limit=1,
                        )
                    )
                    if (
                        terminal_records
                        and observed_terminal_event_id == terminal_records[0].event.id
                    ):
                        return
                elif current.status in _REPLAY_ACTIVE_SESSION_STATUSES:
                    operation_in_progress = True
                    observed_terminal_event_id = None
                else:
                    return
                remaining = idle_deadline - loop.time()
                if remaining <= 0:
                    timeout_error = TimeoutError(
                        f"Replay for session {session_id} received no events for "
                        f"{replay_idle_timeout_s:g} seconds."
                    )
                    yield _stream_error_sse_message(
                        cayu_app,
                        timeout_error,
                        kind="observer",
                        code="replay_idle_timeout",
                        retryable=True,
                        session_id=session_id,
                    )
                    return
                await asyncio.sleep(min(poll_interval, remaining))
                poll_interval = _next_replay_poll_interval(
                    poll_interval,
                    received_events=bool(page),
                )

        return EventSourceResponse(replay(), send_timeout=SSE_SEND_TIMEOUT_SECONDS)

    @bounded_control_plane_router.post(
        "/run",
        response_class=EventSourceResponse,
        responses=RUN_ENDPOINT_RESPONSES,
    )
    async def run_agent(
        body: RunBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
        mutation_id: MutationIdHeader = None,
        auth_context: AuthContext | None = optional_auth_context,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        runtime_generated_session_id = f"session-{uuid4().hex}" if body.session_id is None else None
        session_id = body.session_id or runtime_generated_session_id
        if session_id is None:
            raise AssertionError("Run route did not assign a session identity.")

        if task_store is not None:
            task_id = str(uuid4())

            async def create_run_task(first_event: Event) -> None:
                if first_event.session_id not in {
                    session_id,
                    cayu_app.project_session_id_for_exposure(session_id),
                }:
                    raise RuntimeError(
                        "Run acceptance event belongs to a different session: "
                        f"{first_event.session_id}"
                    )
                snapshot = await session_store.load_invocation_snapshot(session_id)
                if snapshot is None:
                    raise RuntimeError(f"Run session disappeared during acceptance: {session_id}")
                # The runtime is paused at ``first_event`` while this callback
                # executes. COMPLETED/FAILED therefore describe a terminal-prefix
                # stream that needs no task. INTERRUPTING/INTERRUPTED can instead
                # be a concurrent operator request; the task must still be linked
                # so the interrupted session can resume it later.
                if snapshot.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                    return
                redacted_prompt = cayu_app.redact_json(body.prompt)
                if type(redacted_prompt) is not str:
                    raise TypeError("CayuApp prompt redaction must return a string.")
                task_request = task_create_with_runtime_invocation(
                    TaskCreate(
                        task_id=task_id,
                        type="run",
                        title=_bounded_run_task_title(redacted_prompt),
                        session_id=session_id,
                        assigned_agent_name=body.agent,
                        input={"prompt": redacted_prompt},
                    ),
                    source=TaskExecutionSource.HTTP_RUN,
                    session_invocation=snapshot,
                )
                await task_store.create_running_task(
                    task_request,
                    session_invocation=snapshot,
                )

            after_accept = create_run_task
        else:
            task_id = None
            after_accept = None

        request = RunRequest(
            agent_name=body.agent,
            session_id=session_id,
            target=body.target,
            causal_budget_id=body.causal_budget_id,
            task_id=task_id,
            labels=body.labels,
            messages=[Message.text("user", body.prompt)],
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
        )
        runtime_authority_fields = tuple(
            field_name
            for field_name, generated in (
                ("session_id", runtime_generated_session_id is not None),
                ("task_id", task_id is not None),
            )
            if generated
        )
        if runtime_authority_fields:
            request = run_request_with_runtime_generated_authority(
                request,
                *runtime_authority_fields,
            )
        try:
            verified_origin = (
                None
                if auth_context is None
                else InvocationOrigin(
                    trust=InvocationOriginTrust.SERVER_VERIFIED,
                    subject=auth_context.subject,
                    tenant=auth_context.tenant,
                )
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Authenticated identity is not valid durable invocation provenance.",
            ) from exc
        request = run_request_with_runtime_invocation(
            request,
            source=SessionExecutionSource.HTTP_RUN,
            verified_origin=verified_origin,
        )
        return await _accepted_event_stream_response(
            cayu_app.run(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="run",
                session_id=session_id,
                after_accept=after_accept,
            ),
        )

    @bounded_control_plane_router.post(
        "/resume",
        dependencies=protected,
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def resume_agent(
        body: ResumeBody,
        http_request: Request,
        trace_metadata: TraceContextMetadata,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        profile_adoption = None
        if body.profile_adoption is not None:
            requested_by = _request_actor(
                auth_context,
                body.profile_adoption.requested_by,
                field_name="requested_by",
            )
            if requested_by is None:
                raise HTTPException(
                    status_code=400,
                    detail="profile_adoption.requested_by is required when authentication is not configured.",
                )
            profile_adoption = ExecutionProfileAdoptionIntent(
                idempotency_key=body.profile_adoption.idempotency_key,
                reason=body.profile_adoption.reason,
                requested_by=requested_by,
            )

        request = ResumeRequest(
            session_id=body.session_id,
            messages=[Message.text("user", body.prompt)],
            profile_adoption=profile_adoption,
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            metadata=trace_metadata,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )
        request = _with_runtime_resume_transport_metadata(request, trace_metadata)

        return await _accepted_event_stream_response(
            cayu_app.resume(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="resume",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/sessions/{session_id}/compact",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def compact_session(
        session_id: PersistableNonBlankString,
        body: CompactSessionBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        private_session_id = await _resolve_public_session_id(session_id)
        session = await session_store.load(private_session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key=body.idempotency_key,
            expected_run_epoch=body.expected_run_epoch,
            expected_transcript_cursor=body.expected_transcript_cursor,
            instructions=body.instructions,
            limits=body.limits,
            budget_limits=body.budget_limits,
            requested_by=_request_interruption_actor(auth_context, body.requested_by),
        )
        return await _accepted_event_stream_response(
            cayu_app.compact_session(request),
            cayu_app=cayu_app,
            session_id=private_session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="session.compact",
                session_id=private_session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.post(
        "/sessions/{session_id}/messages",
        response_class=EventSourceResponse,
        responses=STREAMING_ENDPOINT_RESPONSES,
    )
    async def enqueue_session_message(
        session_id: NonBlankString,
        body: EnqueueSessionMessageBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        private_session_id = await _resolve_public_session_id(session_id)
        session = await session_store.load(private_session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        async def operation() -> AsyncIterator[Event]:
            result = await cayu_app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key=body.idempotency_key,
                    content=body.content,
                    delivery_mode=body.delivery_mode,
                    requested_by=_request_interruption_actor(
                        auth_context,
                        body.requested_by,
                    ),
                )
            )
            yield result.event

        return await _accepted_event_stream_response(
            operation(),
            cayu_app=cayu_app,
            session_id=private_session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="session.message.enqueue",
                session_id=private_session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/sessions/{session_id}/interrupt",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def interrupt_session(
        session_id: NonBlankString,
        http_request: Request,
        body: InterruptSessionBody | None = None,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=session_id,
        )
        if replay is not None:
            return replay
        private_session_id = await _resolve_public_session_id(session_id)
        session = await session_store.load(private_session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )
        if session.status not in _SERVER_INTERRUPTIBLE_SESSION_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Session cannot be interrupted from status: {session.status.value}",
            )

        request = InterruptSessionRequest(
            session_id=session_id,
            reason=body.reason if body is not None else None,
            metadata=body.metadata if body is not None else {},
            requested_by=_request_interruption_actor(
                auth_context,
                body.requested_by if body is not None else None,
            ),
        )
        return await _accepted_event_stream_response(
            cayu_app.interrupt_session(request),
            cayu_app=cayu_app,
            session_id=private_session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="interrupt",
                session_id=private_session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/provider-operations/resolve",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_provider_operation(
        body: ProviderOperationResolutionBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        if await session_store.load_state(session_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )
        request = ProviderOperationResolutionRequest(
            session_id=body.session_id,
            stage_id=body.stage_id,
            expected_run_epoch=body.expected_run_epoch,
            action=body.action,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
        )
        return await _accepted_event_stream_response(
            cayu_app.resolve_provider_operation(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="provider_operation.resolve",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/tool-approvals/resolve",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_tool_approval(
        body: ToolApprovalBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRequest(
            session_id=body.session_id,
            approval_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.approval_id,
                field_name="approval_id",
            ),
            tool_round_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_round_id,
                field_name="tool_round_id",
            ),
            tool_call_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_call_id,
                field_name="tool_call_id",
            ),
            decision=body.decision,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )

        return await _accepted_event_stream_response(
            cayu_app.resolve_tool_approval(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="tool_approval.resolve",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/tool-approvals/recover",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_approval(
        body: ToolApprovalRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolApprovalRecoveryRequest(
            session_id=body.session_id,
            approval_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.approval_id,
                field_name="approval_id",
            ),
            tool_round_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_round_id,
                field_name="tool_round_id",
            ),
            tool_call_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_call_id,
                field_name="tool_call_id",
            ),
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_tool_approval(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="tool_approval.recover",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/tool-rounds/recover",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_tool_round(
        body: ToolRoundRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = ToolRoundRecoveryRequest(
            session_id=body.session_id,
            round_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.round_id,
                field_name="tool_round_id",
            ),
            tool_call_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_call_id,
                field_name="tool_call_id",
            ),
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_tool_round(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="tool_round.recover",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/user-input/resolve",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def resolve_user_input(
        body: UserInputResolveBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        response = UserInputResponse(
            session_id=body.session_id,
            input_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.input_id,
                field_name="input_id",
            ),
            answer=body.answer,
            structured=body.structured,
            artifacts=body.artifacts,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )

        return await _accepted_event_stream_response(
            cayu_app.resolve_user_input(response),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="user_input.resolve",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @bounded_control_plane_router.post(
        "/user-input/recover",
        response_class=EventSourceResponse,
        responses=BOUNDED_STREAMING_ENDPOINT_RESPONSES,
    )
    async def recover_user_input(
        body: UserInputRecoveryBody,
        http_request: Request,
        auth_context: AuthContext | None = optional_auth_context,
        mutation_id: MutationIdHeader = None,
    ):
        replay = await _replay_events_response(
            http_request,
            expected_session_id=body.session_id,
        )
        if replay is not None:
            return replay
        session_id = await _resolve_public_session_id(body.session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {body.session_id}",
            )

        request = UserInputRecoveryRequest(
            session_id=body.session_id,
            input_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.input_id,
                field_name="input_id",
            ),
            answer=body.answer,
            tool_call_id=await _resolve_public_linkage(
                session_id=session_id,
                value=body.tool_call_id,
                field_name="tool_call_id",
            ),
            outcome=body.outcome,
            message=body.message,
            structured=body.structured,
            artifacts=body.artifacts,
            reason=body.reason,
            metadata=body.metadata,
            resolved_by=_request_resolution_actor(auth_context, body.resolved_by),
            max_steps=body.max_steps,
            limits=body.limits,
            budget_limits=body.budget_limits,
            retry_policy=body.retry_policy,
            structured_output=body.structured_output,
            thinking=body.thinking,
            loop_policies=await _continuation_loop_policies(session_id),
        )

        return await _accepted_event_stream_response(
            cayu_app.recover_user_input(request),
            cayu_app=cayu_app,
            session_id=session_id,
            acceptance_callbacks=_mutation_acceptance_callbacks(
                mutation_id=mutation_id,
                mutation_kind="user_input.recover",
                session_id=session_id,
            ),
            conflict_error_types=(RuntimeError, TimeoutError, ValueError),
        )

    @router.get("/agents", response_model=AgentsResponse, dependencies=protected)
    async def list_agents():
        agents = [
            _serialize_agent(cayu_app, cayu_app.get_agent(name)) for name in cayu_app.list_agents()
        ]
        return {"agents": agents, "total_count": len(agents)}

    @router.get("/agents/{agent_name}", response_model=AgentsResponse, dependencies=protected)
    async def get_agent(agent_name: NonBlankString):
        try:
            agent = cayu_app.get_agent(agent_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent not found") from exc
        serialized = _serialize_agent(cayu_app, agent)
        return {"agents": [serialized], "total_count": 1}

    @router.get(
        "/environments",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def list_environments():
        records = cayu_app.list_environment_registrations()
        environments = [_serialize_environment(cayu_app, record) for record in records]
        return {"environments": environments, "total_count": len(environments)}

    @router.get(
        "/environments/{environment_name}",
        response_model=EnvironmentsResponse,
        dependencies=protected,
    )
    async def get_environment(environment_name: NonBlankString):
        record = next(
            (
                item
                for item in cayu_app.list_environment_registrations()
                if item.spec.name == environment_name
            ),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Environment not found")
        return {"environments": [_serialize_environment(cayu_app, record)], "total_count": 1}

    @router.get(
        "/artifacts",
        response_model=ArtifactsResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def list_artifacts(
        limit: Annotated[int, Query(ge=1, le=_ARTIFACT_PAGE_LIMIT_MAX)] = 100,
        offset: Annotated[int, Query(ge=0, le=_ARTIFACT_PAGE_OFFSET_MAX)] = 0,
        artifact_store_id: Annotated[str | None, Query()] = None,
        scope: ArtifactScope | None = None,
        session_id: Annotated[str | None, Query()] = None,
        agent_name: Annotated[str | None, Query()] = None,
        environment_name: Annotated[str | None, Query()] = None,
    ):
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        requested_agent_name = _clean_optional_query_value(agent_name, "agent_name")
        requested_environment_name = _clean_optional_query_value(
            environment_name,
            "environment_name",
        )
        stores = _artifact_stores_by_id(cayu_app)
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            selected_stores = {requested_store_id: store}
        else:
            selected_stores = stores

        artifacts: list[dict[str, Any]] = []
        total_count: int | None = 0
        truncated = False
        per_store_limit = offset + limit
        for store_id, store in selected_stores.items():
            try:
                result = await store.list(
                    scope=scope,
                    session_id=requested_session_id,
                    agent_name=requested_agent_name,
                    environment_name=requested_environment_name,
                    limit=per_store_limit,
                )
                if type(result) is not ArtifactListResult:
                    raise TypeError("Artifact stores must return ArtifactListResult from list().")
                page = ArtifactListResult(
                    artifacts=result.artifacts,
                    total_count=result.total_count,
                    truncated=result.truncated,
                )
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
            artifacts.extend(
                _serialize_artifact(cayu_app, artifact, artifact_store_id=store_id)
                for artifact in page.artifacts
            )
            if page.total_count is None:
                total_count = None
                truncated = True
            elif total_count is not None:
                total_count += page.total_count
            truncated = truncated or page.truncated

        artifacts.sort(key=_artifact_sort_key, reverse=True)
        page_artifacts = artifacts[offset : offset + limit]
        next_offset = None
        has_more = False
        if total_count is None:
            if len(artifacts) > offset + limit or truncated:
                has_more = True
                candidate_next_offset = offset + limit
                if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                    next_offset = candidate_next_offset
        elif offset + limit < total_count:
            has_more = True
            candidate_next_offset = offset + limit
            if candidate_next_offset <= _ARTIFACT_PAGE_OFFSET_MAX:
                next_offset = candidate_next_offset
        truncated = has_more
        return {
            "artifacts": page_artifacts,
            "total_count": total_count,
            "truncated": truncated,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
        }

    async def _read_artifact_from_request(
        artifact_id: str,
        artifact_store_id: str | None,
        *,
        max_bytes: int | None = None,
    ):
        try:
            artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        stores = _artifact_stores_by_id(cayu_app)
        requested_store_id = _clean_optional_query_value(
            artifact_store_id,
            "artifact_store_id",
        )
        if requested_store_id is not None:
            store = stores.get(requested_store_id)
            if store is None:
                raise HTTPException(status_code=404, detail="Artifact store not found")
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                return requested_store_id, copy_artifact_read_result(
                    read,
                    expected_artifact_id=artifact_id,
                    max_content_bytes=max_bytes,
                )
            except InvalidArtifactIdError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Artifact not found") from exc
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc

        matches = []
        for store_id, store in stores.items():
            try:
                read = await store.read_bytes(artifact_id, max_bytes=max_bytes)
                matches.append(
                    (
                        store_id,
                        copy_artifact_read_result(
                            read,
                            expected_artifact_id=artifact_id,
                            max_content_bytes=max_bytes,
                        ),
                    )
                )
            except (FileNotFoundError, InvalidArtifactIdError):
                continue
            except (ArtifactStoreUnavailableError, OSError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Artifact store is unavailable.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Artifact store returned invalid artifact data.",
                ) from exc
        if not matches:
            raise HTTPException(status_code=404, detail="Artifact not found")
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="Artifact id exists in multiple stores; pass artifact_store_id.",
            )
        return matches[0]

    @router.get(
        "/artifacts/{artifact_id}",
        response_model=ArtifactReadResponse,
        responses=ARTIFACT_ENDPOINT_ERROR_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str | None, Query()] = None,
        max_bytes: Annotated[int, Query(ge=1, le=262_144)] = 64_000,
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        preview_base64, text_preview = _artifact_read_preview(cayu_app, read)
        return {
            "artifact": _serialize_artifact(cayu_app, read.metadata, artifact_store_id=store_id),
            "preview_base64": preview_base64,
            "text_preview": text_preview,
            "total_bytes": read.total_bytes,
            "truncated": read.truncated,
        }

    @router.get(
        "/artifacts/{artifact_id}/content",
        response_class=Response,
        responses=ARTIFACT_CONTENT_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_artifact_content(
        artifact_id: ArtifactIdPath,
        artifact_store_id: Annotated[str, Query(min_length=1)],
        disposition: Annotated[Literal["attachment", "inline"], Query()] = "attachment",
        max_bytes: Annotated[int, Query(ge=1, le=_ARTIFACT_CONTENT_BYTES_MAX)] = (
            _ARTIFACT_CONTENT_BYTES_MAX
        ),
    ):
        store_id, read = await _read_artifact_from_request(
            artifact_id,
            artifact_store_id,
            max_bytes=max_bytes,
        )
        if read.truncated or read.total_bytes > max_bytes or len(read.content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Artifact exceeds the requested max_bytes for direct content "
                    "response. Use the bounded artifact preview, increase max_bytes "
                    "up to the server maximum, or use a store-native streaming/range "
                    "reader for artifacts above that maximum."
                ),
            )
        response_disposition = _artifact_content_disposition_kind(
            read.metadata.content_type,
            disposition,
        )
        return Response(
            content=read.content,
            media_type=read.metadata.content_type,
            headers={
                "Content-Disposition": _artifact_content_disposition(
                    read.metadata.filename,
                    response_disposition,
                ),
                "X-Content-Type-Options": "nosniff",
                "X-Cayu-Artifact-Id": _artifact_header_value(read.metadata.id, artifact_id),
                "X-Cayu-Artifact-Store-Id": _artifact_header_value(store_id, "artifact-store"),
                "Cache-Control": "private, no-store",
            },
        )

    @router.get(
        "/pending-actions",
        response_model=PendingActionsResponse,
        responses=PENDING_ACTION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_pending_actions(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        session_id: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        kind: PendingActionKind | None = None,
        agent_name: Annotated[str | None, Query()] = None,
        environment_name: Annotated[str | None, Query()] = None,
        cursor: Annotated[str | None, Query()] = None,
    ):
        requested_session_id = _clean_optional_query_value(session_id, "session_id")
        if requested_session_id is not None:
            requested_session_id = await _resolve_public_session_id(requested_session_id)
        search = _clean_optional_query_value(q, "q")
        requested_agent_name = _clean_optional_query_value(agent_name, "agent_name")
        requested_environment_name = _clean_optional_query_value(
            environment_name, "environment_name"
        )
        requested_cursor = _clean_optional_query_value(cursor, "cursor")
        if requested_cursor is not None:
            try:
                decode_session_cursor(requested_cursor)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = await cayu_app._runtime_session_store.query_pending_actions(
                PendingActionQuery(
                    session_id=requested_session_id,
                    kind=kind,
                    agent_name=requested_agent_name,
                    environment_name=requested_environment_name,
                    q=search,
                    cursor=requested_cursor,
                    limit=limit,
                )
            )
        except CheckpointCompatibilityError as exc:
            raise HTTPException(
                status_code=409,
                detail=exc.safe_evidence(),
            ) from exc
        except PendingActionResultTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        return {
            "actions": [_serialize_pending_action(cayu_app, action) for action in result.actions],
            "issues": [
                _redact_control_plane_values(
                    cayu_app,
                    issue.model_dump(mode="json"),
                    "pending_action_issue",
                    preserve_string_fields={"code", "status", "updated_at"},
                )
                for issue in result.issues
            ],
            "next_cursor": _serialize_session_cursor(cayu_app, result.next_cursor),
            "has_more": result.has_more,
            "total_count": result.total_count,
            "inspected_candidate_count": result.inspected_candidate_count,
        }

    @router.get("/sessions", response_model=ListSessionsResponse, dependencies=protected)
    async def list_sessions(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        (
            private_parent_session_id,
            private_causal_budget_id,
        ) = await _resolve_session_query_authority_filters(
            parent_session_id=parent_session_id,
            causal_budget_id=causal_budget_id,
        )
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=private_parent_session_id,
                    causal_budget_id=private_causal_budget_id,
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "sessions": [_serialize_session_base(cayu_app, session) for session in result.sessions],
            "next_cursor": _serialize_session_cursor(cayu_app, result.next_cursor),
            "total_count": result.total_count,
        }

    async def get_session_topology(
        session_id: NonBlankString,
        body: SessionTopologyRequest,
        response: Response,
    ):
        expanded_parent_ids = (
            body.expanded_parent_ids if body.expanded_parent_ids else (str(session_id),)
        )
        for parent_id, cursor in body.child_cursors.items():
            try:
                decode_session_topology_cursor(
                    cursor,
                    parent_session_id=parent_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid session topology cursor.",
                ) from exc
        try:
            topology_query = SessionTopologyQuery(
                focus_session_id=str(session_id),
                expanded_parent_ids=expanded_parent_ids,
                child_cursors=body.child_cursors,
                ancestor_depth_limit=body.ancestor_depth_limit,
                child_limit=body.child_limit,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_input=False),
            ) from exc
        observed_at = datetime.now(UTC)
        try:
            result = await session_store.query_session_topology(topology_query)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="The focus session or an expanded parent was not found.",
            ) from exc
        except SessionTopologyDepthExceeded as exc:
            raise HTTPException(
                status_code=413,
                detail=("The focus session's ancestry exceeds the requested ancestor_depth_limit."),
            ) from exc
        except SessionTopologyCycle as exc:
            raise HTTPException(
                status_code=409,
                detail="The focus session's durable ancestry contains a cycle.",
            ) from exc
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail="The configured session store does not support topology queries.",
            ) from exc
        except ValidationError as exc:
            raise HTTPException(
                status_code=409,
                detail="The session store returned an inconsistent topology projection.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail="The focus session's durable ancestry is inconsistent.",
            ) from exc

        _require_safe_session_topology_authority(cayu_app, result)
        loaded_session_ids = {
            result.focus.id,
            *(node.id for node in result.ancestors),
            *(node.id for node in result.expanded_parents),
            *(child.id for branch in result.branches for child in branch.children),
        }
        linked_task_session_ids = (
            body.linked_task_session_ids if body.linked_task_session_ids else (result.focus.id,)
        )
        if set(linked_task_session_ids).difference(loaded_session_ids):
            raise HTTPException(
                status_code=422,
                detail=(
                    "linked_task_session_ids may contain only sessions loaded by "
                    "this topology request."
                ),
            )
        if set(body.task_session_cursors).difference(linked_task_session_ids):
            raise HTTPException(
                status_code=422,
                detail=("task_session_cursors keys must identify selected task-linked sessions."),
            )
        if set(body.task_child_cursors).difference(body.expanded_task_parent_ids):
            raise HTTPException(
                status_code=422,
                detail=("task_child_cursors keys must identify expanded task parents."),
            )
        for session_id_with_cursor, cursor in body.task_session_cursors.items():
            try:
                decode_task_topology_cursor(
                    cursor,
                    scope_kind="session",
                    scope_id=session_id_with_cursor,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid task topology cursor.",
                ) from exc
        for parent_task_id, cursor in body.task_child_cursors.items():
            try:
                decode_task_topology_cursor(
                    cursor,
                    scope_kind="parent_task",
                    scope_id=parent_task_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid task topology cursor.",
                ) from exc
        try:
            task_query = TaskTopologyQuery(
                linked_session_ids=linked_task_session_ids,
                session_cursors=body.task_session_cursors,
                expanded_parent_ids=body.expanded_task_parent_ids,
                child_cursors=body.task_child_cursors,
                session_task_limit=body.task_session_limit,
                child_limit=body.task_child_limit,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_input=False),
            ) from exc

        task_result: TaskTopologyStoreResult | None = None
        if task_store is None:
            task_projection_status: Literal["available", "not_configured", "unsupported"] = (
                "not_configured"
            )
        elif not task_store.supports_task_topology:
            task_projection_status = "unsupported"
        else:
            try:
                task_result = await task_store.query_task_topology(task_query)
                task_result.validate_for_query(task_query)
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="A requested expanded task parent was not found.",
                ) from exc
            except TaskTopologyCycle as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The loaded durable task topology contains a cycle.",
                ) from exc
            except TaskTopologyTraversalLimitExceeded as exc:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Task topology ancestry exceeds the server's bounded validation limits."
                    ),
                ) from exc
            except (TaskTopologyInconsistent, ValidationError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The task store returned an inconsistent topology projection.",
                ) from exc
            except NotImplementedError:
                task_projection_status = "unsupported"
            else:
                task_projection_status = "available"
                _require_safe_task_topology_authority(cayu_app, task_result)

        serialized_branches = []
        for branch in result.branches:
            next_cursor = _serialize_session_topology_cursor(
                cayu_app,
                branch.next_cursor,
                parent_session_id=branch.parent_session_id,
            )
            serialized_branches.append(
                {
                    "parent_session_id": _redact_control_plane_values(
                        cayu_app,
                        {"parent_session_id": branch.parent_session_id},
                        "session_topology.branch",
                    )["parent_session_id"],
                    "children": [
                        _serialize_session_topology_node(cayu_app, child)
                        for child in branch.children
                    ],
                    "next_cursor": next_cursor,
                    "has_more": branch.has_more,
                }
            )

        unique_node_ids = {
            result.focus.id,
            *(node.id for node in result.ancestors),
            *(node.id for node in result.expanded_parents),
            *(child.id for branch in result.branches for child in branch.children),
        }
        response_value = SessionTopologyResponse.model_validate(
            {
                "observed_at": observed_at,
                "cross_store_atomic": False,
                "focus": _serialize_session_topology_node(cayu_app, result.focus),
                "ancestors": [
                    _serialize_session_topology_node(cayu_app, node) for node in result.ancestors
                ],
                "expanded_parents": [
                    _serialize_session_topology_node(cayu_app, node)
                    for node in result.expanded_parents
                ],
                "branches": serialized_branches,
                "unique_node_count": len(unique_node_ids),
                "task_projection": _serialize_task_topology_projection(
                    cayu_app,
                    task_result,
                    status=task_projection_status,
                ),
                "edges": _execution_topology_edges(result, task_result),
            }
        )
        response_payload = response_value.model_dump(mode="json")
        if not json_utf8_size_within_limit(
            response_payload,
            body.max_result_bytes,
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "Session topology exceeds max_result_bytes. Request fewer "
                    "expanded session/task branches or smaller branch limits."
                ),
            )
        response.headers["Cache-Control"] = "private, no-store"
        return response_payload

    router.add_api_route(
        "/sessions/{session_id}/topology",
        get_session_topology,
        methods=["POST"],
        response_model=SessionTopologyResponse,
        responses=SESSION_TOPOLOGY_ENDPOINT_RESPONSES,
        dependencies=protected,
        route_class_override=_BoundedSessionTopologyRoute,
    )

    @router.post(
        "/sessions/summary",
        response_model=SessionsSummaryResponse,
        dependencies=protected,
    )
    async def get_sessions_summary(
        body: SessionsSummaryBody | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        offset: Annotated[int, Query(ge=0)] = 0,
        cursor: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        status: SessionStatus | None = None,
        debug_state: SessionDebugState | None = None,
        agent_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        environment_name: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
    ):
        body = body or SessionsSummaryBody()
        labels = _parse_session_label_filters(label)
        label_selectors = _parse_session_label_selectors(label_selector)
        (
            private_parent_session_id,
            private_causal_budget_id,
        ) = await _resolve_session_query_authority_filters(
            parent_session_id=parent_session_id,
            causal_budget_id=causal_budget_id,
        )
        try:
            result = await session_store.list_sessions(
                SessionQuery(
                    q=_clean_optional_query_value(q, "q"),
                    status=status,
                    debug_state=debug_state,
                    agent_name=_clean_optional_query_value(agent_name, "agent_name"),
                    provider_name=_clean_optional_query_value(provider_name, "provider_name"),
                    model=_clean_optional_query_value(model, "model"),
                    environment_name=_clean_optional_query_value(
                        environment_name,
                        "environment_name",
                    ),
                    parent_session_id=private_parent_session_id,
                    causal_budget_id=private_causal_budget_id,
                    labels=labels,
                    label_selectors=label_selectors,
                    limit=limit,
                    offset=offset,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sessions = result.sessions
        session_event_records_by_id: dict[str, list[EventRecord]] = {}
        session_ids = [session.id for session in sessions]
        all_event_records = await _query_all_session_event_records(session_ids)
        for record in all_event_records:
            session_event_records_by_id.setdefault(record.event.session_id, []).append(record)
        for session in sessions:
            session_event_records_by_id.setdefault(session.id, [])

        usage_event_records = [
            record
            for record in all_event_records
            if record.event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}
        ]
        usage_events = [
            record.event
            for record in sorted(usage_event_records, key=lambda record: record.sequence)
        ]
        usage_summary = _serialize_causal_budget_usage_summary(
            cayu_app,
            causal_budget_usage_summary(
                causal_budget_id="session-query",
                session_ids=session_ids,
                events=usage_events,
            ),
        )
        usage_summary.pop("causal_budget_id", None)
        model_events = [
            record.event
            for record in sorted(all_event_records, key=lambda record: record.sequence)
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        provider_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, None),
        )
        model_breakdown = _usage_breakdown(
            model_events,
            key_fn=lambda metrics: (metrics.provider_name, metrics.model),
        )

        cost_summary = None
        if body.pricing is not None:
            aggregate_cost = build_session_cost_summary(
                session_id="session-query",
                events=model_events,
                pricing=body.pricing,
                currency=body.currency,
            ).model_dump(mode="json")
            aggregate_cost.pop("session_id", None)
            aggregate_cost["session_ids"] = [
                cayu_app.project_session_id_for_exposure(session_id) for session_id in session_ids
            ]
            aggregate_cost["session_count"] = len(session_ids)
            aggregate_cost["session_costs"] = [
                _serialize_session_cost_summary(
                    cayu_app,
                    build_session_cost_summary(
                        session_id=session.id,
                        events=[
                            record.event
                            for record in session_event_records_by_id[session.id]
                            if record.event.type == EventType.MODEL_COMPLETED
                        ],
                        pricing=body.pricing,
                        currency=body.currency,
                    ),
                )
                for session in sessions
            ]
            cost_summary = aggregate_cost

        session_items = []
        for session in sessions:
            records = session_event_records_by_id[session.id]
            outcome = session_outcome_from_records(session, records)
            event_summary = event_summary_from_records(session.id, records)
            session_items.append(
                {
                    "session": _serialize_session(cayu_app, session),
                    "outcome": _serialize_session_outcome(cayu_app, outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(cayu_app, event_summary.latest_event)
                        ),
                    },
                }
            )

        return {
            "session_count": len(sessions),
            "sessions": session_items,
            "next_cursor": _serialize_session_cursor(cayu_app, result.next_cursor),
            "total_count": result.total_count,
            "usage": usage_summary,
            "provider_breakdown": provider_breakdown,
            "model_breakdown": model_breakdown,
            "cost": cost_summary,
        }

    @router.get(
        "/sessions/{session_id}/usage",
        response_model=SessionUsageSummary,
        dependencies=protected,
    )
    async def get_session_usage(session_id: NonBlankString):
        try:
            summary = await cayu_app.get_session_usage(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary

    @router.post(
        "/sessions/{session_id}/cost",
        response_model=SessionCostSummary,
        dependencies=protected,
    )
    async def estimate_session_cost(session_id: NonBlankString, body: SessionCostBody):
        try:
            summary = await cayu_app.get_session_cost(
                session_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary

    @router.get(
        "/causal-budgets/{causal_budget_id}/usage",
        response_model=CausalBudgetUsageSummary,
        dependencies=protected,
    )
    async def get_causal_budget_usage(causal_budget_id: NonBlankString):
        try:
            summary = await cayu_app.get_causal_budget_usage(causal_budget_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return summary

    @router.post(
        "/causal-budgets/{causal_budget_id}/cost",
        response_model=CausalBudgetCostSummary,
        dependencies=protected,
    )
    async def estimate_causal_budget_cost(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        try:
            summary = await cayu_app.get_causal_budget_cost(
                causal_budget_id,
                body.pricing,
                currency=body.currency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Causal budget not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return summary

    @router.post(
        "/causal-budgets/{causal_budget_id}/summary",
        response_model=CausalBudgetSummaryResponse,
        responses=CAUSAL_BUDGET_SUMMARY_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_causal_budget_summary(
        causal_budget_id: NonBlankString,
        body: SessionCostBody,
    ):
        try:
            private_causal_budget_id = await cayu_app._resolve_public_causal_budget_id(
                causal_budget_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        sessions = await _list_all_causal_sessions(
            private_causal_budget_id,
            max_sessions=_CAUSAL_BUDGET_SUMMARY_MAX_SESSIONS,
        )
        if not sessions:
            raise HTTPException(status_code=404, detail="Causal budget not found")

        session_ids = [session.id for session in sessions]
        causal_event_records = await _query_all_causal_event_records(
            private_causal_budget_id,
            max_events=_CAUSAL_BUDGET_SUMMARY_MAX_EVENTS,
            max_bytes=_CAUSAL_BUDGET_SUMMARY_MAX_EVENT_INPUT_BYTES,
        )
        event_records_by_session_id: dict[str, list[EventRecord]] = {
            session_id: [] for session_id in session_ids
        }
        for record in causal_event_records:
            event_records_by_session_id.setdefault(record.event.session_id, []).append(record)
        usage_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.MODEL_COMPLETED
        ]
        tool_event_records = [
            record
            for record in causal_event_records
            if record.event.type == EventType.TOOL_CALL_STARTED
        ]
        usage_events = [
            record.event
            for record in sorted(
                [*usage_event_records, *tool_event_records],
                key=lambda record: record.sequence,
            )
        ]
        usage_summary = causal_budget_usage_summary(
            causal_budget_id=private_causal_budget_id,
            session_ids=session_ids,
            events=usage_events,
        )
        cost_summary = build_causal_budget_cost_summary(
            causal_budget_id=private_causal_budget_id,
            session_ids=session_ids,
            events=[record.event for record in usage_event_records],
            pricing=body.pricing,
            currency=body.currency,
        )
        session_items = []
        for session in sessions:
            session_event_records = event_records_by_session_id[session.id]
            outcome = session_outcome_from_records(
                session,
                session_event_records,
            )
            event_summary = event_summary_from_records(
                session.id,
                session_event_records,
            )
            session_items.append(
                {
                    "session": _serialize_session(cayu_app, session),
                    "outcome": _serialize_session_outcome(cayu_app, outcome),
                    "events": {
                        "total_events": event_summary.total_events,
                        "counts_by_type": event_summary.counts_by_type,
                        "latest_event": (
                            None
                            if event_summary.latest_event is None
                            else _serialize_event_record(cayu_app, event_summary.latest_event)
                        ),
                    },
                }
            )

        response_value = {
            "causal_budget_id": cayu_app.project_causal_budget_id_for_exposure(
                private_causal_budget_id,
                session_ids=(session.id for session in sessions),
            ),
            "session_count": len(sessions),
            "sessions": session_items,
            "usage": _serialize_causal_budget_usage_summary(cayu_app, usage_summary),
            "cost": _serialize_causal_budget_cost_summary(cayu_app, cost_summary),
        }
        if not json_utf8_size_within_limit(
            response_value,
            _CAUSAL_BUDGET_SUMMARY_MAX_RESULT_BYTES,
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "Causal-budget summary exceeds max_result_bytes. Use the bounded "
                    "session topology and store-native usage rollup for large workflows."
                ),
            )
        return response_value

    @router.get(
        "/sessions/{session_id}/state",
        response_model=SessionStateResponse,
        dependencies=protected,
    )
    async def get_session_state(session_id: NonBlankString):
        session_id = await _resolve_public_session_id(session_id)
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        interruption_cascade = await cayu_app.interruption_cascade_status(session_id)
        provider_operation = await inspect_provider_operation(session_store, session_id)
        return {
            "session_id": cayu_app.project_session_id_for_exposure(state.id),
            "status": state.status,
            "updated_at": state.updated_at.isoformat(),
            "last_activity_at": state.last_activity_at.isoformat(),
            "interruption_cascade": interruption_cascade,
            "provider_operation": provider_operation.model_dump(mode="json"),
        }

    @router.get(
        "/sessions/{session_id}/summary",
        response_model=SessionSummaryResponse,
        dependencies=protected,
    )
    async def get_session_summary(session_id: NonBlankString):
        session_id = await _resolve_public_session_id(session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        event_summary = await session_store.summarize_events(session_id)
        outcome = await session_store.summarize_outcome(session_id)
        transcript_page = await session_store.query_transcript(
            TranscriptQuery(session_id=session_id, limit=1)
        )
        usage_summary = await cayu_app.get_session_usage(session_id)

        return {
            "session": _serialize_session(cayu_app, session),
            "events": {
                "total_events": event_summary.total_events,
                "counts_by_type": event_summary.counts_by_type,
                "latest_event": (
                    None
                    if event_summary.latest_event is None
                    else _serialize_event_record(cayu_app, event_summary.latest_event)
                ),
            },
            "transcript": {
                "total_messages": transcript_page.total_records,
            },
            "outcome": _serialize_session_outcome(cayu_app, outcome),
            "usage": usage_summary,
        }

    async def _list_all_causal_sessions(
        causal_budget_id: str,
        *,
        max_sessions: int,
    ) -> list[Session]:
        sessions: list[Session] = []
        cursor = None
        while True:
            remaining_with_sentinel = max_sessions + 1 - len(sessions)
            result = await session_store.list_sessions(
                SessionQuery(
                    causal_budget_id=causal_budget_id,
                    limit=min(1000, remaining_with_sentinel),
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
            page = result.sessions
            if not page:
                return sessions
            sessions.extend(page)
            if len(sessions) > max_sessions:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Causal-budget summary exceeds the {max_sessions}-session "
                        "safety limit. Use the bounded session topology."
                    ),
                )
            if result.next_cursor is None:
                return sessions
            cursor = result.next_cursor

    async def _query_all_session_event_records(session_ids: list[str]) -> list[EventRecord]:
        if not session_ids:
            return []
        records: list[EventRecord] = []
        after_sequence = None
        while True:
            page = await session_store.query_events(
                EventQuery(
                    session_ids=tuple(session_ids),
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < 5000:
                return records
            after_sequence = page[-1].sequence

    async def _query_all_causal_event_records(
        causal_budget_id: str,
        *,
        max_events: int,
        max_bytes: int,
    ) -> list[EventRecord]:
        records: list[EventRecord] = []
        after_sequence = None
        remaining_bytes = max_bytes
        while True:
            if remaining_bytes <= 0:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Causal-budget summary exceeds the {max_bytes}-byte "
                        "event-input safety limit. Use the bounded session topology "
                        "and store-native usage rollup."
                    ),
                )
            remaining_with_sentinel = max_events + 1 - len(records)
            try:
                page = await session_store.query_events_bounded(
                    EventQuery(
                        causal_budget_id=causal_budget_id,
                        after_sequence=after_sequence,
                        limit=min(5000, remaining_with_sentinel),
                    ),
                    max_bytes=remaining_bytes,
                )
            except EventQueryResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Causal-budget summary exceeds the {max_bytes}-byte "
                        "event-input safety limit. Use the bounded session topology "
                        "and store-native usage rollup."
                    ),
                ) from exc
            except NotImplementedError as exc:
                raise HTTPException(
                    status_code=501,
                    detail=(
                        "The configured session store cannot enforce byte-bounded "
                        "causal-budget event reads."
                    ),
                ) from exc
            if not page:
                return records
            size_counter = JsonUtf8SizeCounter(remaining_bytes)
            for record in page:
                if not size_counter.value(record):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Causal-budget summary exceeds the {max_bytes}-byte "
                            "event-input safety limit. Use the bounded session "
                            "topology and store-native usage rollup."
                        ),
                    )
            remaining_bytes = size_counter.remaining
            records.extend(page)
            if len(records) > max_events:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Causal-budget summary exceeds the {max_events}-event "
                        "safety limit. Use the bounded session topology and "
                        "store-native usage rollup."
                    ),
                )
            if len(page) < min(5000, remaining_with_sentinel):
                return records
            after_sequence = page[-1].sequence

    @router.get(
        "/sessions/{session_id}/interactions",
        response_model=ListSessionInteractionsResponse,
        dependencies=protected,
    )
    async def list_session_interactions(
        session_id: NonBlankString,
        before_sequence: int | None = Query(default=None, ge=1),
        limit: int = Query(default=50, ge=1, le=100),
    ):
        session_id = await _resolve_public_session_id(session_id)
        public_session_id = cayu_app.project_session_id_for_exposure(session_id)
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        records = await session_store.query_latest_interaction_events(
            session_id,
            before_sequence=before_sequence,
            limit=limit + 1,
        )
        page = records[:limit]
        return {
            "session_id": public_session_id,
            "interactions": [_serialize_interaction_record(cayu_app, record) for record in page],
            "next_sequence": page[-1].sequence if page else before_sequence,
            "has_more": len(records) > limit,
        }

    @router.get(
        "/sessions/{session_id}/interactions/{interaction_id}",
        response_model=ApiInteractionSummary,
        dependencies=protected,
    )
    async def get_session_interaction(
        session_id: NonBlankString,
        interaction_id: NonBlankString,
    ):
        session_id = await _resolve_public_session_id(session_id)
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        interaction_id = await _resolve_public_interaction_id(
            session_id=session_id,
            value=interaction_id,
        )
        records = await session_store.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            raise HTTPException(status_code=404, detail="Interaction not found")
        return _serialize_interaction_record(cayu_app, records[0])

    @router.get(
        "/sessions/{session_id}/events",
        response_model=ListSessionEventsResponse,
        dependencies=protected,
    )
    async def list_session_events(
        session_id: NonBlankString,
        event_id: str | None = Query(
            default=None,
            description="Return the event with this exact session-scoped event ID, if it exists.",
        ),
        event_type: str | None = None,
        interaction_id: str | None = Query(
            default=None,
            description="Return only events attributed to this interaction.",
        ),
        exclude_event_type: str | None = Query(
            default=None,
            description="Exclude one event type before applying pagination.",
        ),
        tool_name: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        workflow_name: str | None = None,
        after_sequence: int | None = Query(
            default=None,
            ge=0,
            description="Return only events with a greater durable sequence.",
        ),
        before_sequence: int | None = Query(
            default=None,
            ge=1,
            description="Return only events with a smaller durable sequence.",
        ),
        order_by: Annotated[
            EventOrder,
            Query(description="Return events in durable sequence order."),
        ] = EventOrder.SEQUENCE_ASC,
        limit: int = Query(default=100, ge=1, le=_EVENT_PAGE_LIMIT_MAX),
    ):
        session_id = await _resolve_public_session_id(session_id)
        public_session_id = cayu_app.project_session_id_for_exposure(session_id)
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if interaction_id is not None:
            interaction_id = await _resolve_public_interaction_id(
                session_id=session_id,
                value=interaction_id,
            )

        private_event_id = event_id
        unresolved_public_event_id = False
        if event_id is not None:
            resolved_record, handled = await _public_or_legacy_event_record(
                session_id,
                event_id,
            )
            if handled:
                if resolved_record is None:
                    private_event_id = None
                    unresolved_public_event_id = True
                else:
                    private_event_id = resolved_record.event.id

        has_event_filters = any(
            value is not None
            for value in (
                event_id,
                event_type,
                interaction_id,
                exclude_event_type,
                tool_name,
                agent_name,
                environment_name,
                workflow_name,
            )
        )
        try:
            query = EventQuery(
                session_id=session_id,
                event_id=private_event_id,
                event_type=event_type,
                interaction_id=interaction_id,
                exclude_event_types=(exclude_event_type,) if exclude_event_type is not None else (),
                tool_name=tool_name,
                agent_name=agent_name,
                environment_name=environment_name,
                workflow_name=workflow_name,
                after_sequence=after_sequence,
                before_sequence=before_sequence,
                limit=limit + 1,
                order_by=order_by,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_context=False, include_url=False),
            ) from exc

        raw_scan_sequence: int | None = None
        if before_sequence is None and has_event_filters:
            # Capture the raw high-water mark before the filtered read. Advancing
            # through a sequence observed after that read could skip a matching
            # event committed between the two queries.
            latest_records = await session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            if latest_records:
                raw_scan_sequence = latest_records[0].sequence

        records = [] if unresolved_public_event_id else await session_store.query_events(query)
        page = records[:limit]
        has_more = len(records) > limit
        cursor = after_sequence if order_by == EventOrder.SEQUENCE_ASC else before_sequence
        next_sequence = page[-1].sequence if page else cursor

        scan_through_sequence: int | None = None
        if before_sequence is None:
            scan_candidates = [sequence for sequence in (after_sequence,) if sequence is not None]
            if order_by == EventOrder.SEQUENCE_ASC and has_more:
                # A matching event remains beyond this response, so the tail
                # cursor must stop at the last event actually returned.
                if page:
                    scan_candidates.append(page[-1].sequence)
            else:
                scan_candidates.extend(record.sequence for record in page)
                if raw_scan_sequence is not None:
                    scan_candidates.append(raw_scan_sequence)
            if scan_candidates:
                scan_through_sequence = max(scan_candidates)

        return {
            "session_id": public_session_id,
            "events": [_serialize_event_record(cayu_app, record) for record in page],
            "order_by": order_by,
            "next_sequence": next_sequence,
            "scan_through_sequence": scan_through_sequence,
            "has_more": has_more,
        }

    @router.get(
        "/sessions/{session_id}/transcript",
        response_model=SessionTranscriptResponse,
        dependencies=protected,
    )
    async def get_session_transcript(
        session_id: NonBlankString,
        role: MessageRole | None = None,
        interaction_id: Annotated[
            NonBlankString | None,
            Query(description="Return only transcript records attributed to this interaction."),
        ] = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=_TRANSCRIPT_PAGE_LIMIT_MAX),
        include_thinking: bool = Query(default=True),
    ):
        session_id = await _resolve_public_session_id(session_id)
        public_session_id = cayu_app.project_session_id_for_exposure(session_id)
        state = await session_store.load_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if interaction_id is not None:
            interaction_id = await _resolve_public_interaction_id(
                session_id=session_id,
                value=interaction_id,
            )

        transcript_page = await session_store.query_transcript(
            TranscriptQuery(
                session_id=session_id,
                role=role,
                interaction_id=interaction_id,
                offset=offset,
                limit=limit,
                include_thinking=include_thinking,
            )
        )
        # Advance by the queried window size, not the returned record count: the
        # include_thinking filter can drop thinking-only records from a page, so
        # len(records) under-counts the messages consumed and would stall pagination.
        consumed = min(limit, max(0, transcript_page.total_records - offset))
        next_offset = offset + consumed

        return {
            "session_id": public_session_id,
            "messages": [
                _serialize_transcript_message(
                    cayu_app,
                    session_id,
                    record.index,
                    record.message,
                    record.interaction_id,
                )
                for record in transcript_page.records
            ],
            "offset": offset,
            "next_offset": next_offset,
            "has_more": next_offset < transcript_page.total_records,
            "total_messages": transcript_page.total_records,
        }

    @router.get(
        "/sessions/{session_id}",
        response_model=ApiSessionDetail,
        dependencies=protected,
    )
    async def get_session(session_id: NonBlankString):
        session_id = await _resolve_public_session_id(session_id)
        session = await session_store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _serialize_session_detail(cayu_app, session)

    @router.delete("/sessions/{session_id}", status_code=204, dependencies=protected)
    async def delete_session(session_id: NonBlankString):
        session_id = await _resolve_public_session_id(session_id)
        try:
            if not await cayu_app.discard_parked_egress_allocations(session_id):
                raise ValueError(
                    "Session has a parked egress allocation whose cleanup remains in flight."
                )
            await session_store.delete_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return None

    @router.patch(
        "/sessions/{session_id}/labels",
        dependencies=protected,
        response_model=ApiSession,
    )
    async def update_session_labels(
        session_id: NonBlankString,
        body: UpdateSessionLabelsBody,
    ):
        session_id = await _resolve_public_session_id(session_id)
        try:
            require_durable_json_text(body.labels, "labels")
            session = await session_store.update_labels(session_id, body.labels)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(cayu_app, session)

    @router.patch(
        "/sessions/{session_id}/metadata",
        dependencies=protected,
        response_model=ApiSession,
    )
    async def update_session_metadata(
        session_id: NonBlankString,
        body: UpdateSessionMetadataBody,
    ):
        session_id = await _resolve_public_session_id(session_id)
        try:
            require_durable_json_text(body.metadata, "metadata")
            session = await session_store.update_metadata(session_id, body.metadata)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialize_session(cayu_app, session)

    @router.get("/tasks", response_model=list[ApiTaskListItem], dependencies=protected)
    async def list_tasks(
        q: str | None = None,
        status: TaskStatus | None = None,
        task_type: str | None = Query(default=None, alias="type"),
        session_id: str | None = None,
        parent_task_id: str | None = None,
        assigned_agent_name: str | None = None,
        order_by: TaskOrder = TaskOrder.UPDATED_AT_DESC,
        limit: int = 50,
        offset: int = 0,
    ):
        if task_store is None:
            return []
        try:
            query = TaskQuery(
                q=q,
                status=status,
                type=task_type,
                session_id=session_id,
                parent_task_id=parent_task_id,
                assigned_agent_name=assigned_agent_name,
                order_by=order_by,
                limit=limit,
                offset=offset,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        tasks = await task_store.list_tasks(query)
        return [_serialize_task_list_item(cayu_app, t) for t in tasks]

    async def _require_task_store():
        if task_store is None:
            raise HTTPException(status_code=404, detail="Task store is not configured.")
        return task_store

    @router.get(
        "/tasks/{task_id}",
        response_model=ApiTaskDetail,
        dependencies=protected,
    )
    async def get_task(task_id: NonBlankString):
        store = await _require_task_store()
        task = await store.load_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return _serialize_task_detail(cayu_app, task)

    async def _apply_task_action(action, task_id: str):
        try:
            task = await action(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_task_detail(cayu_app, task)

    @router.post(
        "/tasks/{task_id}/pause",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def pause_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.pause_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/block",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def block_task(task_id: NonBlankString, body: TaskHoldBody | None = None):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.block_task(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/needs-attention",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def mark_task_needs_attention(
        task_id: NonBlankString,
        body: TaskHoldBody | None = None,
    ):
        store = await _require_task_store()
        request_body = body or TaskHoldBody()
        return await _apply_task_action(
            lambda task_id: store.mark_task_needs_attention(
                task_id,
                reason=request_body.reason,
                payload=request_body.payload,
            ),
            task_id,
        )

    @router.post(
        "/tasks/{task_id}/resume",
        dependencies=protected,
        response_model=ApiTaskDetail,
    )
    async def resume_task(task_id: NonBlankString):
        store = await _require_task_store()
        return await _apply_task_action(store.resume_task, task_id)

    def _knowledge_review_workflow() -> KnowledgeReviewWorkflow:
        if knowledge_store is None:
            raise HTTPException(status_code=404, detail="Knowledge store is not configured.")
        return KnowledgeReviewWorkflow(
            knowledge_store,
            access_scope=knowledge_access_scope,
            namespace=knowledge_review_namespace,
            labels=knowledge_review_labels,
            default_limit=50,
        )

    async def _apply_knowledge_review_action(action, entry_id: str):
        try:
            entry = await action(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KnowledgeRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _serialize_reviewed_knowledge_entry(entry)

    @router.get(
        "/knowledge/pending",
        response_model=PendingKnowledgeListResponse,
        dependencies=protected,
    )
    async def list_pending_knowledge(
        namespace: str | None = None,
        label: Annotated[list[str] | None, Query()] = None,
        kind: Annotated[list[str] | None, Query()] = None,
        aspect: Annotated[list[str] | None, Query()] = None,
        visibility: Annotated[list[KnowledgeVisibility] | None, Query()] = None,
        source_type: str | None = None,
        source_id: str | None = None,
        limit: int = 50,
        max_bytes: int = 20_000,
    ):
        workflow = _knowledge_review_workflow()
        try:
            result = await workflow.list_pending(
                namespace=_clean_optional_query_value(namespace, "namespace"),
                labels=_parse_knowledge_label_filters(label),
                kinds=_parse_knowledge_string_filters(kind, "kind") if kind is not None else None,
                visibilities=visibility,
                aspects=_parse_knowledge_string_filters(aspect, "aspect"),
                source_type=_clean_optional_query_value(source_type, "source_type"),
                source_id=_clean_optional_query_value(source_id, "source_id"),
                limit=limit,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "entries": [_serialize_knowledge_list_item(item) for item in result.entries],
            "truncated": result.truncated,
            "limit": result.limit,
            "max_bytes": result.max_bytes,
            "total_entries_known": result.total_entries_known,
        }

    @router.get(
        "/knowledge/pending/{entry_id}",
        response_model=PendingKnowledgeDetailResponse,
        dependencies=protected,
    )
    async def get_pending_knowledge(
        entry_id: NonBlankString,
        max_chunks: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_CHUNKS,
        max_bytes: Annotated[
            int,
            Query(ge=1, le=_KNOWLEDGE_PENDING_DETAIL_MAX_BYTES),
        ] = _KNOWLEDGE_PENDING_DETAIL_MAX_BYTES,
    ):
        workflow = _knowledge_review_workflow()
        assert knowledge_store is not None
        try:
            entry = await workflow.get_pending(entry_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            chunks = await knowledge_store.read_chunks(
                entry.id,
                revision=entry.revision,
                access_scope=knowledge_access_scope,
                max_chunks=max_chunks,
                max_bytes=max_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            **_serialize_knowledge_detail(entry),
            "chunks": [_serialize_knowledge_chunk(chunk) for chunk in chunks],
            "chunk_limit": max_chunks,
            "chunk_max_bytes": max_bytes,
        }

    @router.post(
        "/knowledge/{entry_id}/approve",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def approve_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.approve, entry_id)

    @router.post(
        "/knowledge/{entry_id}/reject",
        dependencies=protected,
        response_model=ApiReviewedKnowledgeEntry,
    )
    async def reject_knowledge(entry_id: NonBlankString):
        workflow = _knowledge_review_workflow()
        return await _apply_knowledge_review_action(workflow.reject, entry_id)

    @router.get("/health", response_model=HealthResponse)
    async def health():
        return {"ok": True}

    router.include_router(bounded_control_plane_router)
    router.include_router(bounded_evaluation_promotion_router)
    router.include_router(bounded_captured_evaluation_router)
    router.include_router(bounded_evals_router)
    return router
