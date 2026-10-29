"""Application-registered evaluators that portable corpora address by key."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from cayu._validation import compact_json_utf8_size, json_utf8_size_within_limit
from cayu.applications import CayuApp
from cayu.evals.assertions import EvalAssertion
from cayu.evals.corpus import (
    _APPLICATION_EVALUATOR_RESULT_METADATA_KEY,
    _RESERVED_RESULT_METADATA_PREFIX,
    EVAL_APPLICATION_EVALUATOR_MAX_MESSAGE_BYTES,
    EVAL_APPLICATION_EVALUATOR_MAX_MESSAGE_CHARS,
    EVAL_APPLICATION_EVALUATOR_MAX_METADATA_BYTES,
    ApplicationEvaluatorAssertionSpec,
    EvaluationEvidencePolicySpec,
    _model_python_input,
    _portable_id,
    _sha256_revision,
    assertion_spec_revision,
)
from cayu.evals.evidence import _redacted_text, _validated_policy
from cayu.evals.json_subset import copy_eval_tool_json_object
from cayu.evals.models import (
    EvalAssertionResult,
    EvalContext,
    EvalOutcome,
    ProbeRequirements,
    _model_instance_python_input,
)

ApplicationEvaluatorFactory: TypeAlias = Callable[[Mapping[str, Any]], EvalAssertion]

_SCORED_OUTCOMES = frozenset({EvalOutcome.PASSED, EvalOutcome.FAILED})


class ApplicationEvaluator(BaseModel):
    """Trusted assertion factory that a portable corpus addresses by key.

    A corpus case names ``key`` and supplies bounded JSON parameters. The trusted
    target calls ``factory`` with a copy of those parameters while compiling the
    corpus. ``implementation_revision`` is the application's declared identity for
    the factory and the assertions it builds; published results retain it so
    comparisons reject results produced by different evaluator implementations.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    key: StrictStr
    implementation_revision: StrictStr
    factory: ApplicationEvaluatorFactory = Field(exclude=True, repr=False)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("implementation_revision")
    @classmethod
    def validate_implementation_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


def _without_reserved_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if not key.startswith(_RESERVED_RESULT_METADATA_PREFIX)
    }


def _copy_application_evaluator(evaluator: ApplicationEvaluator) -> ApplicationEvaluator:
    if type(evaluator) is not ApplicationEvaluator:
        raise TypeError("evaluator must be an exact ApplicationEvaluator.")
    return ApplicationEvaluator(
        key=evaluator.key,
        implementation_revision=evaluator.implementation_revision,
        factory=evaluator.factory,
    )


class _CompiledApplicationEvaluatorAssertion(EvalAssertion):
    """One portable assertion bound to the trusted application evaluator it names."""

    _app: CayuApp
    _assertion_revision: str
    _evaluator: ApplicationEvaluator
    _evidence_policy: EvaluationEvidencePolicySpec
    _inner: EvalAssertion
    _spec: ApplicationEvaluatorAssertionSpec

    __slots__ = (
        "_app",
        "_assertion_revision",
        "_evaluator",
        "_evidence_policy",
        "_inner",
        "_spec",
    )

    def __init__(
        self,
        spec: ApplicationEvaluatorAssertionSpec,
        *,
        evaluator: ApplicationEvaluator,
        app: CayuApp,
        evidence_policy: EvaluationEvidencePolicySpec,
    ) -> None:
        validated_spec = ApplicationEvaluatorAssertionSpec.model_validate(_model_python_input(spec))
        validated_evaluator = _copy_application_evaluator(evaluator)
        if validated_spec.evaluator_key != validated_evaluator.key:
            raise ValueError("Portable application evaluator key does not match its binding.")
        if not isinstance(app, CayuApp):
            raise TypeError("app must be a CayuApp.")
        params = copy_eval_tool_json_object(validated_spec.params, "params")
        try:
            inner = validated_evaluator.factory(params)
        except Exception as exc:
            raise ValueError(
                f"Application evaluator {validated_evaluator.key!r} could not build "
                f"assertion {validated_spec.id!r} ({type(exc).__name__})."
            ) from exc
        if inspect.iscoroutine(inner):
            inner.close()
        if not isinstance(inner, EvalAssertion):
            raise ValueError(
                f"Application evaluator {validated_evaluator.key!r} must build an "
                f"EvalAssertion for assertion {validated_spec.id!r}."
            )
        object.__setattr__(self, "_spec", validated_spec)
        object.__setattr__(self, "_assertion_revision", assertion_spec_revision(validated_spec))
        object.__setattr__(self, "_evaluator", validated_evaluator)
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_app", app)
        # The shared evidence view also provides the trial's public output preview.
        object.__setattr__(self, "_evidence_policy", _validated_policy(evidence_policy))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Compiled application evaluator assertions are immutable.")

    @property
    def name(self) -> str:
        return self._spec.id

    @property
    def assertion_revision(self) -> str:
        return self._assertion_revision

    @property
    def evaluates_failed_session(self) -> bool:
        return self._inner.evaluates_failed_session

    @property
    def reads_final_output_only(self) -> bool:
        return self._inner.reads_final_output_only

    def required_probes(self) -> ProbeRequirements:
        return self._inner.required_probes()

    def _with_public_contract(self, result: EvalAssertionResult) -> EvalAssertionResult:
        """Replace reserved metadata with the trusted evaluator identity."""

        metadata = _without_reserved_metadata(result.metadata)
        metadata[_APPLICATION_EVALUATOR_RESULT_METADATA_KEY] = {
            "evaluator_key": self._evaluator.key,
            "implementation_revision": self._evaluator.implementation_revision,
        }
        return result.model_copy(update={"metadata": metadata})

    def _public_result(self, result: EvalAssertionResult, app: CayuApp) -> EvalAssertionResult:
        """Bind one evaluator result to the corpus identity and ``app``'s public bounds."""

        metadata = _without_reserved_metadata(result.metadata)
        message = _redacted_text(app, result.message, "application evaluator message")
        redacted_metadata = app.redact_json(metadata)
        if type(redacted_metadata) is not dict:
            raise ValueError("Application evaluator metadata redaction must preserve its object.")
        if result.outcome in _SCORED_OUTCOMES:
            if (
                len(message) > EVAL_APPLICATION_EVALUATOR_MAX_MESSAGE_CHARS
                or compact_json_utf8_size(message) > EVAL_APPLICATION_EVALUATOR_MAX_MESSAGE_BYTES
            ):
                return self._with_public_contract(
                    self.error("Application evaluator message exceeded its public bound.")
                )
            try:
                redacted_metadata = copy_eval_tool_json_object(redacted_metadata, "metadata")
            except (TypeError, ValueError):
                redacted_metadata = None
            if redacted_metadata is None or not json_utf8_size_within_limit(
                redacted_metadata,
                EVAL_APPLICATION_EVALUATOR_MAX_METADATA_BYTES,
                ensure_ascii=False,
            ):
                return self._with_public_contract(
                    self.error("Application evaluator metadata exceeded its public bound.")
                )
        return self._with_public_contract(
            EvalAssertionResult(
                name=self.name,
                assertion_revision=self.assertion_revision,
                outcome=result.outcome,
                score=result.score,
                threshold=result.threshold,
                message=message,
                metadata=redacted_metadata,
                cost_summary=result.cost_summary,
            )
        )

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        return await self.evaluate_in(context, runtime_app=None)

    async def evaluate_in(
        self,
        context: EvalContext,
        *,
        runtime_app: CayuApp | None,
    ) -> EvalAssertionResult:
        """Evaluate within the redaction boundary of the app that ran the trial.

        A fresh trial's ``runtime_app`` owns its redaction boundary, as it does for
        the shared assertion evidence. The app retained at compilation is used only
        when there is no live runtime app, as on replay.
        """

        app = self._app if runtime_app is None else runtime_app
        if not isinstance(app, CayuApp):
            raise TypeError("runtime_app must be a CayuApp or None.")
        try:
            result = self._inner.evaluate(context)
            if inspect.isawaitable(result):
                result = await result
            if type(result) is not EvalAssertionResult:
                raise TypeError("EvalAssertion.evaluate must return EvalAssertionResult.")
            # Rebuild inside this boundary so a validator-bypassed result cannot
            # reach publication.
            result = EvalAssertionResult.model_validate(_model_instance_python_input(result))
            # Checked here too, so an invalid result keeps the evaluator's identity.
            if result.cost_summary is not None and (
                context.session is None or result.cost_summary.session_id != context.session.id
            ):
                raise ValueError("Assertion cost summaries must belong to the trajectory session.")
            return self._public_result(result, app)
        except Exception as exc:
            return self._with_public_contract(
                self.error(f"Application evaluator raised {type(exc).__name__}.")
            )
