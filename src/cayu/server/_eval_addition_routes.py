"""Authenticated preview, atomic publication and exact receipt recovery for session additions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Header, HTTPException

from cayu.evals.promotion import SessionPromotionError
from cayu.evals.read_views import decode_suite_view
from cayu.evals.session_additions import (
    EvalAdditionCommitRequest,
    EvalAdditionConflict,
    EvalAdditionPlan,
    EvalAdditionReceipt,
    EvalAdditionRecord,
    EvalAdditionRequest,
    EvalAdditionSourceLink,
    addition_operation_id,
    addition_revision,
)
from cayu.evals.store import (
    EvalAuthoredSuiteConflict,
    EvalStore,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server._captured_evaluation_source import CapturedEvaluationSource
from cayu.server._eval_addition_planner import EvalAdditionPlanner
from cayu.server._eval_view_reader import EvalViewReader
from cayu.server.auth import AuthContext
from cayu.server.contracts import CAPTURED_EVALUATION_ENDPOINT_RESPONSES

if TYPE_CHECKING:
    from fastapi.params import Depends


def register_eval_addition_routes(
    router: APIRouter,
    *,
    store: EvalStore | None,
    source: CapturedEvaluationSource,
    optional_auth_context: Depends,
) -> None:
    request_auth = cast("AuthContext | None", optional_auth_context)

    def require_store() -> EvalStore:
        if store is None or not store.session_additions:
            raise HTTPException(status_code=409, detail="Session additions are unavailable.")
        return store

    async def prepare(session_id: str, body: EvalAdditionRequest):
        try:
            return await EvalAdditionPlanner(source, require_store()).prepare(session_id, body)
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413, detail="The session addition exceeds the byte limit."
            ) from exc
        except SessionPromotionError as exc:
            raise HTTPException(
                status_code=409, detail="This session cannot be converted to a runnable case."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid session addition.") from exc

    @router.post(
        "/evals/sessions/{session_id}/additions/preview",
        response_model=EvalAdditionPlan,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    )
    async def preview_session_addition(
        session_id: str,
        body: EvalAdditionRequest,
        auth_context: AuthContext | None = request_auth,
    ) -> EvalAdditionPlan:
        del auth_context  # The dependency authenticates and applies the current role policy.
        return (await prepare(session_id, body)).plan

    @router.post(
        "/evals/sessions/{session_id}/additions",
        response_model=EvalAdditionReceipt,
        status_code=201,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    )
    async def publish_session_addition(
        session_id: str,
        body: EvalAdditionCommitRequest,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=512)
        ],
        auth_context: AuthContext | None = request_auth,
    ) -> EvalAdditionReceipt:
        active_store = require_store()
        subject = None if auth_context is None else auth_context.subject
        tenant = None if auth_context is None else auth_context.tenant
        operation_id = addition_operation_id(subject, tenant, idempotency_key)
        # This complete request includes all expected source/plan revisions. A digest
        # is supplementary evidence, never a substitute for exact replay equality.
        material = {"session_id": session_id, "request": body.model_dump(mode="json")}
        previous = await active_store.load_session_addition(operation_id)
        if previous is not None:
            if (
                previous.request_material != material
                or previous.receipt.request_revision != addition_revision(material)
                or previous.subject != subject
                or previous.tenant != tenant
            ):
                raise HTTPException(
                    status_code=409,
                    detail="The operation key is already bound to another addition.",
                )
            # Reauthorization precedes recovery; source mutation/deletion must not
            # erase an already committed publication or force it to run again.
            return previous.receipt
        request = EvalAdditionRequest.model_validate(
            body.model_dump(
                mode="python", exclude={"expected_plan_revision", "expected_source_revision"}
            )
        )
        prepared = await prepare(session_id, request)
        if (
            prepared.plan.revision != body.expected_plan_revision
            or prepared.plan.source_revision != body.expected_source_revision
        ):
            raise HTTPException(
                status_code=409, detail="The reviewed addition changed; preview it again."
            )
        record = EvalAdditionRecord(
            receipt=EvalAdditionReceipt(
                operation_id=operation_id,
                request_revision=addition_revision(material),
                plan=prepared.plan,
                created_at=datetime.now(UTC),
            ),
            subject=subject,
            tenant=tenant,
            public_session_id=session_id,
            private_session_id=prepared.private_session_id,
            request_material=material,
        )
        try:
            saved = await active_store.publish_session_addition(
                prepared, record, redact_json=source.app.redact_json
            )
        except (EvalAdditionConflict, EvalAuthoredSuiteConflict) as exc:
            raise HTTPException(
                status_code=409, detail="The session addition conflicts with its stored operation."
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422, detail="The session addition contains unsafe public data."
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413, detail="The session addition exceeds the byte limit."
            ) from exc
        return saved.receipt

    @router.get(
        "/evals/additions/{operation_id}",
        response_model=EvalAdditionReceipt,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    )
    async def get_session_addition(
        operation_id: str,
        auth_context: AuthContext | None = request_auth,
    ) -> EvalAdditionReceipt:
        record = await require_store().load_session_addition(operation_id)
        subject = None if auth_context is None else auth_context.subject
        tenant = None if auth_context is None else auth_context.tenant
        if record is None or record.subject != subject or record.tenant != tenant:
            raise HTTPException(status_code=404, detail="Session addition not found.")
        return record.receipt

    @router.get(
        "/evals/suite-views/{suite_key}/cases/{case_id}/source",
        response_model=EvalAdditionSourceLink,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    )
    async def get_addition_source(
        suite_key: str,
        case_id: str,
        result_revision: str | None = None,
        auth_context: AuthContext | None = request_auth,
    ) -> EvalAdditionSourceLink:
        active_store = require_store()
        unavailable = EvalAdditionSourceLink(session_id=None, state="unavailable")
        try:
            identity = decode_suite_view(suite_key)
            _, cases = await EvalViewReader(active_store, source.registry).suite_document(identity)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid source-link scope.") from exc
        record = (
            await active_store.load_case_addition(suite_key, case_id)
            if result_revision is None
            else await active_store.load_result_addition(result_revision, suite_key)
        )
        if record is None:
            return unavailable
        plan = record.receipt.plan
        # Membership lookup owns the exact child association; its receipt retains
        # the original publication identity. Score placement remains exact.
        if plan.case_id != case_id or (
            result_revision is not None and plan.destination != identity
        ):
            return unavailable
        if result_revision is None and not any(
            case.id == case_id and case.revision == plan.case_revision for case in cases
        ):
            return unavailable
        if result_revision is not None and plan.captured_result_revision != result_revision:
            return unavailable
        # A stored server-owned source link must still resolve to the same live
        # source. Portable case source fields never enter this decision.
        try:
            resolved = await source.resolve_session(record.public_session_id)
        except HTTPException:
            return unavailable
        if (
            resolved != record.private_session_id
            or await source.app.session_store.load(resolved) is None
        ):
            return unavailable
        return EvalAdditionSourceLink(session_id=record.public_session_id, state="available")
