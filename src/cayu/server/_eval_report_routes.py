"""Reporting-only publication; operator authentication never substitutes for a grant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.utils import create_model_field

from cayu.evals.reported import (
    EVAL_REPORT_JSON_LIMITS,
    EVAL_REPORT_MAX_BYTES,
    EvalReport,
    EvalReportConflict,
    EvalReportReceipt,
)
from cayu.evals.store import EvalStorePublicationRejected, EvalStoreResultTooLarge
from cayu.server._http_json import _validated_private_json_body
from cayu.server.auth import AuthContext
from cayu.server.contracts import EVALS_ENDPOINT_RESPONSES
from cayu.server.eval_report_auth import (
    EvalReporterBearerAuth,
    EvalReportingAccess,
    EvalReportPrincipal,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from fastapi.routing import APIRoute

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry

_PRINCIPAL_SCOPE_KEY = "cayu.eval-report.principal"


def reporting_router(
    bounded_route: type[APIRoute],
    *,
    store: EvalStore | None,
    registry: EvalTargetRegistry | None,
    access: EvalReportingAccess | None,
    cayu_app: Any,
) -> APIRouter:
    authenticate = (
        access.authenticate if access is not None else EvalReporterBearerAuth.from_environment()
    )

    async def report_auth(request: Request) -> AuthContext:
        try:
            headers = request.headers.getlist("authorization")
            if len(headers) > 1 or any(len(header.encode("utf-8")) > 4096 for header in headers):
                raise ValueError("Oversized reporting credential.")
            principal = None if authenticate is None else await authenticate(request)
            if type(principal) is EvalReportPrincipal:
                principal = EvalReportPrincipal.model_validate(
                    principal.model_dump(mode="python", warnings=False)
                )
        except Exception:
            principal = None
        if type(principal) is not EvalReportPrincipal:
            raise HTTPException(status_code=401, detail="Evals reporting authentication required.")
        # Revalidate trusted extension output before copying authority into request scope.
        request.scope[_PRINCIPAL_SCOPE_KEY] = principal
        return AuthContext(subject=principal.publisher_id)

    class ReportRoute(bounded_route):  # ty: ignore[unsupported-base] -- receives the existing bounded APIRoute owner.
        max_request_bytes = EVAL_REPORT_MAX_BYTES
        invalid_request_detail = "Invalid eval report."
        oversized_request_detail = "Eval report exceeds the server byte limit."
        reject_duplicate_json_keys = True
        preparse_auth = staticmethod(report_auth)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.body_field is not None:
                raise RuntimeError("Eval report route must retain private body parsing ownership.")
            # As with MemoryReportRoute, runtime parsing stays with the bounded
            # owner; this field publishes the reusable OpenAPI input component.
            self.body_field = create_model_field(
                name=f"Body_{self.unique_id}",
                type_=EvalReport,
                field_info=Body(annotation=EvalReport),
                mode="validation",
            )

    router = APIRouter(route_class=ReportRoute)

    @router.post(
        "/evals/reports",
        response_model=EvalReportReceipt,
        responses=EVALS_ENDPOINT_RESPONSES,
        description="Publish immutable external evaluation evidence using a dedicated reporting credential and target grant. Operator authentication alone does not authorize this endpoint.",
    )
    async def publish_report(request: Request):
        report = await _validated_private_json_body(
            request,
            EvalReport,
            invalid_detail="Invalid eval report.",
            json_mode=True,
            json_limits=EVAL_REPORT_JSON_LIMITS,
        )
        principal = request.scope.get(_PRINCIPAL_SCOPE_KEY)
        if type(principal) is not EvalReportPrincipal:
            raise HTTPException(status_code=401, detail="Evals reporting authentication required.")
        if report.target_key not in principal.target_keys:
            raise HTTPException(status_code=403, detail="Reporting target is not authorized.")
        if store is None or not store.reports or registry is None:
            raise HTTPException(status_code=409, detail="Evals reporting is unavailable.")
        if registry.get(report.target_key) is None:
            raise HTTPException(status_code=404, detail="Reporting target is unavailable.")
        # The incoming credential is not part of the app secret registry. Reject
        # echoes of it as well as configured workload secrets before persistence.
        header = request.headers.get("authorization", "")
        credential = header.partition(" ")[2]
        credential_redactor = SecretRedactor(credential or None)

        def redact(value):
            return credential_redactor.redact_json(cayu_app.redact_json(value))

        try:
            record = await store.publish_report(
                report, publisher_id=principal.publisher_id, redact_json=redact
            )
        except EvalReportConflict:
            raise HTTPException(
                status_code=409, detail="Eval report conflicts with stored evidence."
            ) from None
        except EvalStoreResultTooLarge:
            raise HTTPException(
                status_code=413, detail="Eval report exceeds the server byte limit."
            ) from None
        except (EvalStorePublicationRejected, ValueError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid or unsafe eval report.") from None
        return record.receipt

    return router
