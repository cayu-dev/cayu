"""Safe structured summaries in the existing report error-string contract."""

from __future__ import annotations

import json
from typing import Literal

from cayu._validation import (
    canonical_durable_json_bytes,
    extract_durable_value_error,
    safe_durable_value_error_bounds,
    safe_durable_value_error_details,
)
from cayu.evals.capture_policy import WorkflowCaptureStage
from cayu.failure_evidence import FailureEvidence, exception_evidence


def workflow_exception_diagnostic(
    exc: BaseException,
    *,
    phase: Literal["execution", "evidence_preparation"],
    stage: WorkflowCaptureStage,
) -> tuple[str, FailureEvidence | None]:
    """Never stringify exception content or let projection replace a failure.

    The JSON suffix is versioned independently of the report: old readers still
    consume an ordinary error string. Caller-supplied durable field labels are
    omitted; numeric ordinal paths and bounds use the existing safe helpers.
    """
    try:
        name = type.__dict__["__name__"].__get__(type(exc), type)
        if (
            type(name) is not str
            or len(name) > 128
            or not name.isascii()
            or not name.isidentifier()
        ):
            name = "Exception"
    except BaseException:
        name = "Exception"
    diagnostic: dict[str, object] = {
        "version": 1,
        "phase": phase,
        "stage": stage,
        "exception_type": name,
        "projection": "available",
    }
    evidence = None
    try:
        evidence = exception_evidence(exc)
        # Revalidate even extension-mutated evidence before serialization.
        snapshot = FailureEvidence.model_validate(evidence.model_dump(mode="python"))
        diagnostic["failure_evidence"] = snapshot.model_dump(mode="json")
    except BaseException:
        diagnostic["projection"] = "unavailable"
        evidence = None
    try:
        durable = extract_durable_value_error(exc)
        if durable is not None:
            code, path = safe_durable_value_error_details(durable)
            limit, observed = safe_durable_value_error_bounds(durable)
            diagnostic["durable_value"] = {
                "code": code,
                "path": path,
                "limit": limit,
                "observed_lower_bound": observed,
            }
    except BaseException:
        diagnostic["projection"] = "unavailable"
    prefix = (
        "Workflow execution failed"
        if phase == "execution"
        else "Workflow eval evidence preparation failed"
    )
    try:
        encoded = canonical_durable_json_bytes(
            diagnostic, "workflow exception diagnostic", max_bytes=65536, max_nodes=4096
        )
    except BaseException:
        # Bad or oversized projected metadata must not hide the original type.
        diagnostic.pop("failure_evidence", None)
        diagnostic["projection"] = "unavailable"
        evidence = None
        encoded = json.dumps(diagnostic, separators=(",", ":")).encode("utf-8")
    return prefix + ": " + encoded.decode("utf-8"), evidence
