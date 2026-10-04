"""Shared HTTP evaluation invocation, preparation, retry and admission functions."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from fastapi import HTTPException

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    require_clean_nonblank,
    require_unicode_scalar_text,
)
from cayu.budgets.run_limits import RunLimits
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.corpus import EvalCorpusDocument, eval_suite_trial_policy
from cayu.evals.execution import CompiledCorpusSuite, CorpusTarget, compile_corpus_suite
from cayu.evals.store import (
    EvalRunAdmissionConflict,
    EvalRunCostBudget,
    EvalRunInvocation,
    EvalRunRecord,
    EvalRunRequest,
    EvalScenarioRunInvocation,
    EvalStorePublicationRejected,
)
from cayu.evals.trial_policy import EvalSuiteRunExposureV1
from cayu.server.auth import AuthContext
from cayu.server.evals_registry import target_for_eval_invocation
from cayu.sessions.invocation import InvocationOrigin, InvocationOriginTrust, SessionExecutionSource

if TYPE_CHECKING:
    from pydantic import BaseModel

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


_EVAL_ADMISSION_REQUEST_REVISION_DOMAIN = b"cayu-eval-admission-request-v1\0"


def _eval_target(target_key: str | None = None, *, active_eval_registry: EvalTargetRegistry):
    selected_key = active_eval_registry.default_target_key if target_key is None else target_key
    target = active_eval_registry.get(selected_key)
    if target is None:
        raise HTTPException(status_code=404, detail="Eval target not found.")
    return target


def eval_run_invocation(
    auth_context: AuthContext | None,
    *,
    max_steps: int | None,
    limits: RunLimits | None,
    cost_budget: EvalRunCostBudget | None,
    scenario: EvalScenarioRunInvocation | None = None,
    authored_suite_revision: str | None = None,
    authored_suite_selection_revision: str | None = None,
    authored_suite_launch_revision: str | None = None,
    authored_suite_launch_lane: int | None = None,
    authored_suite_exposure: EvalSuiteRunExposureV1 | None = None,
) -> EvalRunInvocation:
    """Build validated HTTP execution bounds and authenticated provenance."""

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
            authored_suite_revision=authored_suite_revision,
            authored_suite_selection_revision=(authored_suite_selection_revision),
            authored_suite_launch_revision=authored_suite_launch_revision,
            authored_suite_launch_lane=authored_suite_launch_lane,
            authored_suite_exposure=authored_suite_exposure,
            scenario=scenario,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Eval execution bounds or authenticated provenance are invalid.",
        ) from exc


def bind_eval_admission_request(
    invocation: EvalRunInvocation,
    *,
    kind: Literal["authored_suite", "captured", "corpus", "scenario"],
    target_key: str,
    resource_identity: Mapping[str, object],
    body: BaseModel,
) -> EvalRunInvocation:
    """Bind a retry to its resource, request and authenticated provenance."""

    material = {
        "kind": kind,
        "target_key": target_key,
        "resource_identity": copy_json_value(
            dict(resource_identity),
            "eval admission resource identity",
        ),
        "request": body.model_dump(mode="json"),
        "invocation_provenance": {
            "source": invocation.source.value,
            "origin": (
                None if invocation.origin is None else invocation.origin.model_dump(mode="json")
            ),
        },
    }
    revision = (
        "sha256:"
        + hashlib.sha256(
            _EVAL_ADMISSION_REQUEST_REVISION_DOMAIN
            + canonical_durable_json_bytes(material, "eval admission request")
        ).hexdigest()
    )
    return EvalRunInvocation.model_validate(
        {
            **invocation.model_dump(mode="python"),
            "admission_request_revision": revision,
        }
    )


def eval_idempotency_digest(
    target_key: str,
    idempotency_key: str,
    *,
    namespace: str | None = None,
) -> str:
    """Keep persisted retry keys scoped to their target and optional launch lane."""

    try:
        clean_key = require_clean_nonblank(idempotency_key, "Idempotency-Key")
        require_unicode_scalar_text(clean_key, "Idempotency-Key")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key.") from exc
    if namespace is None:
        domain = b"cayu-server-eval-idempotency-v1\0"
    else:
        try:
            clean_namespace = require_clean_nonblank(
                namespace,
                "internal eval idempotency namespace",
            )
            namespace_bytes = clean_namespace.encode("ascii")
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            raise RuntimeError("Invalid internal eval idempotency namespace.") from exc
        domain = b"cayu-server-eval-idempotency-v1\0internal\0" + namespace_bytes + b"\0"
    return (
        "sha256:"
        + hashlib.sha256(
            domain + target_key.encode("ascii") + b"\0" + clean_key.encode("utf-8")
        ).hexdigest()
    )


async def replay_eval_run(
    *,
    eval_store: EvalStore,
    target_key: str,
    idempotency_key: str,
    admission_request_revision: str,
    idempotency_namespace: str | None = None,
) -> EvalRunRecord | None:
    """Return an accepted run only when its request binding still matches."""

    digest = eval_idempotency_digest(
        target_key,
        idempotency_key,
        namespace=idempotency_namespace,
    )
    existing = await eval_store.load_run_by_idempotency_key(digest)
    if existing is None:
        return None
    if (
        existing.spec.target_key != target_key
        or existing.spec.invocation.admission_request_revision != admission_request_revision
    ):
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key is already bound to another eval run request.",
        )
    return existing


async def admit_eval_run(
    *,
    eval_store: EvalStore,
    corpus: EvalCorpusDocument,
    max_concurrency: int,
    invocation: EvalRunInvocation,
    idempotency_key: str,
    idempotency_namespace: str | None = None,
    eval_target: CorpusTarget,
    compiled: CompiledCorpusSuite,
) -> EvalRunRecord:
    """Persist a prepared request through the target redaction boundary."""

    if not eval_store.trial_checkpointing:
        raise HTTPException(
            status_code=409,
            detail="Restart-safe eval trial checkpointing is not available.",
        )
    if eval_target.key != corpus.target_key:
        raise RuntimeError("Prepared eval target does not match its corpus.")
    if invocation.execution_profile is None:
        raise RuntimeError("Server-admitted eval run lost its execution-profile binding.")
    if invocation.admission_request_revision is None:
        raise RuntimeError("Server-admitted eval run lost its admission request revision.")
    run_request = EvalRunRequest(
        run_id=f"eval-{uuid4().hex}",
        corpus_revision=corpus.revision,
        target_key=eval_target.key,
        suite_id=compiled.run_contract.suite_id,
        suite_revision=compiled.run_contract.suite_revision,
        max_concurrency=max_concurrency,
        invocation=invocation,
        idempotency_key=eval_idempotency_digest(
            eval_target.key,
            idempotency_key,
            namespace=idempotency_namespace,
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


async def prepare_eval_run(
    *,
    active_eval_registry: EvalTargetRegistry,
    corpus: EvalCorpusDocument,
    suite_id: str,
    max_concurrency: int,
    invocation: EvalRunInvocation,
    expected_execution_profile_revision: str | None = None,
    expect_exact_execution_profile: bool = False,
) -> tuple[CorpusTarget, CompiledCorpusSuite, EvalRunInvocation]:
    """Validate bounds and current profiles before compiling fresh work."""

    eval_target = _eval_target(corpus.target_key, active_eval_registry=active_eval_registry)
    registration = active_eval_registry.registration(eval_target.key)
    if registration is None:
        raise HTTPException(status_code=404, detail="Eval target not found.")
    suite = next((item for item in corpus.suites if item.id == suite_id), None)
    policy = registration.execution_profile_policy
    if suite is not None and suite.trial_request.trials > policy.max_trials:
        raise HTTPException(
            status_code=400,
            detail="Eval run exceeds the published execution-profile trial limit.",
        )
    if suite is not None and max_concurrency > eval_suite_trial_policy(suite).max_concurrency:
        raise HTTPException(
            status_code=400,
            detail="Eval run exceeds the immutable suite concurrency policy.",
        )
    if max_concurrency > policy.max_concurrency:
        raise HTTPException(
            status_code=400,
            detail=("Eval run exceeds the published execution-profile concurrency limit."),
        )
    if any(case.suite_id == suite_id and case.input is None for case in corpus.cases):
        raise HTTPException(
            status_code=409,
            detail=(
                "This captured evaluation has no runnable input. Author runnable "
                "input or a scenario before launching fresh work."
            ),
        )
    try:
        published_profile_revision = None
        if expected_execution_profile_revision is not None and not expect_exact_execution_profile:
            published_profile = await active_eval_registry.prepare_execution_profile(
                eval_target.key
            )
            published_profile_revision = published_profile.snapshot.revision
            if published_profile.snapshot.revision != expected_execution_profile_revision:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "The selected eval execution profile changed after it was "
                        "reviewed. Refresh readiness before launching."
                    ),
                )
    except HTTPException:
        raise
    except EvalExecutionProfileChangedError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The application identity for this eval execution profile changed after "
                "the target was published. Refresh the deployment before launching."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail="The current eval execution profile is unavailable.",
        ) from exc
    try:
        effective_target = target_for_eval_invocation(
            registration.execution_target(),
            invocation,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Eval run is incompatible with the attached target or bounds.",
        ) from exc
    try:
        prepared_profile = await active_eval_registry.prepare_execution_profile(
            eval_target.key,
            effective_target=effective_target,
        )
        effective_target = prepared_profile.target
        if (
            expected_execution_profile_revision is not None
            and expect_exact_execution_profile
            and prepared_profile.snapshot.revision != expected_execution_profile_revision
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "The exact eval execution profile changed after readiness. "
                    "Check launch readiness again."
                ),
            )
        invocation = invocation.model_copy(
            update={
                "execution_profile": prepared_profile.binding,
                "execution_profile_snapshot": prepared_profile.snapshot,
            },
            deep=True,
        )
        if published_profile_revision is not None:
            current_published_profile = await active_eval_registry.prepare_execution_profile(
                eval_target.key
            )
            if current_published_profile.snapshot.revision != published_profile_revision:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "The selected eval execution profile changed during launch "
                        "preparation. Refresh readiness before launching."
                    ),
                )
    except HTTPException:
        raise
    except EvalExecutionProfileChangedError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The application identity for this eval execution profile changed after "
                "the target was published. Refresh the deployment before launching."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail="The exact current eval execution profile is unavailable.",
        ) from exc
    try:
        compiled = await asyncio.to_thread(
            compile_corpus_suite,
            corpus,
            effective_target,
            suite_id,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Eval corpus is incompatible with the attached target or bounds.",
        ) from exc
    return effective_target, compiled, invocation
