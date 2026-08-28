from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from cayu.evals.assertions import EvalAssertion
from cayu.evals.corpus import (
    _MODEL_JUDGE_RESOLVED_IMPLEMENTATION_REVISION_METADATA_KEY,
    _STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY,
    EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
    AssertionSpec,
    EvaluationEvidencePolicySpec,
    JudgePrivacyPolicyV1,
    JudgeProfileIdentityV1,
    MaxEstimatedCostAssertionSpec,
    ModelJudgeAssertionSpec,
    PricingProfileIdentityV1,
    PrivateJudgeReferenceV1,
    PublicJudgeReferenceV1,
    RootStatusAssertionSpec,
    StructuredModelJudgeAssertionSpec,
    _bounded_durable_text,
    _content_revision,
    _model_python_input,
    _portable_id,
    _validated_assertion_spec,
    assertion_spec_revision,
    pricing_profile_identity,
)
from cayu.evals.evidence import (
    AssertionEvidenceView,
    _build_assertion_evidence_view,
    _redacted_text,
    _validated_currencies,
    _validated_policy,
    _validated_pricing,
    _ValidatedPricingSnapshot,
)
from cayu.evals.judges import (
    LLMJudge,
    StructuredLLMJudge,
    _first_user_text,
    _isolated_structured_judge_app,
    _render_transcript,
)
from cayu.evals.memory_attribution import EvalMemoryAttributionEvidenceV1
from cayu.evals.models import EvalAssertionResult, EvalContext
from cayu.evals.portable_evaluation import _evaluate_validated_assertion_spec
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, copy_price_book
from cayu.runtime.manifest import AppManifest

MODEL_JUDGE_EXECUTION_SEMANTICS_VERSION = 1


@dataclass(frozen=True, slots=True)
class _CompiledPricingBinding:
    """Compile-time pricing identity backed by one shared trusted source."""

    source: PriceBook | None
    fingerprint: str | None


_NO_PRICING = _CompiledPricingBinding(source=None, fingerprint=None)


@dataclass(frozen=True, slots=True)
class _TrustedModelJudgeBinding:
    """Trusted execution authority resolved outside the portable corpus."""

    key: str
    app: CayuApp
    agent_name: str
    implementation_revision: str
    profile: JudgeProfileIdentityV1 | None = None
    privacy_policy: JudgePrivacyPolicyV1 | None = None
    private_references: tuple[_TrustedPrivateJudgeReferenceBinding, ...] = ()
    price_book: PriceBook | None = None
    candidate_route_relation: str = "independent_model"
    structured_app: CayuApp | None = None


@dataclass(frozen=True, slots=True)
class _TrustedPrivateJudgeReferenceBinding:
    """Non-portable evaluator truth retained behind one exact public identity."""

    key: str
    revision: str
    content: str
    privacy_policy_key: str
    privacy_policy_revision: str


def _model_judge_implementation_revision(
    *,
    key: str,
    app: CayuApp,
    agent_name: str,
) -> str:
    """Fingerprint judge semantics plus the trusted application's public manifest."""

    validated_key = _portable_id(key, "key")
    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    validated_agent_name = _bounded_durable_text(
        agent_name,
        "agent_name",
        max_chars=256,
        nonblank=True,
        clean=True,
    )
    registered_agent = app.get_agent(validated_agent_name)
    if registered_agent.tools or registered_agent.hosted_tools:
        raise ValueError("Trusted model judge agents must be registered without tools.")
    manifest = app.describe()
    if type(manifest) is not AppManifest:
        raise TypeError("CayuApp.describe() must return an AppManifest.")
    agent_manifest = next(
        (agent for agent in manifest.agents if agent.name == validated_agent_name),
        None,
    )
    if agent_manifest is None:
        raise ValueError("Trusted model judge agent is absent from the application manifest.")
    if agent_manifest.resolved_provider is None:
        raise ValueError("Trusted model judge agent must resolve exactly one provider.")
    app.get_provider(agent_manifest.resolved_provider)
    redacted_agent_spec = app.redact_json(registered_agent.spec.model_dump(mode="json"))
    if type(redacted_agent_spec) is not dict:
        raise TypeError("Trusted model judge agent redaction must preserve its object shape.")
    agent_spec_revision = _content_revision(
        redacted_agent_spec,
        "trusted model judge agent specification",
    )
    return _content_revision(
        {
            "model_judge_execution_semantics_version": (MODEL_JUDGE_EXECUTION_SEMANTICS_VERSION),
            "evaluator_key": validated_key,
            "agent_name": validated_agent_name,
            "agent_spec_revision": agent_spec_revision,
            "app_manifest_schema_version": manifest.schema_version,
            "app_manifest_fingerprint": manifest.fingerprint,
        },
        "trusted model judge implementation",
    )


def _trusted_model_judge_binding(
    *,
    key: str,
    app: CayuApp,
    agent_name: str,
    profile: JudgeProfileIdentityV1 | None = None,
    privacy_policy: JudgePrivacyPolicyV1 | None = None,
    private_references: Sequence[_TrustedPrivateJudgeReferenceBinding] = (),
    price_book: PriceBook | None = None,
    candidate_route_relation: str = "independent_model",
) -> _TrustedModelJudgeBinding:
    validated_key = _portable_id(key, "key")
    validated_agent_name = _bounded_durable_text(
        agent_name,
        "agent_name",
        max_chars=256,
        nonblank=True,
        clean=True,
    )
    if profile is not None and type(profile) is not JudgeProfileIdentityV1:
        raise TypeError("profile must be an exact JudgeProfileIdentityV1 or None.")
    if privacy_policy is not None and type(privacy_policy) is not JudgePrivacyPolicyV1:
        raise TypeError("privacy_policy must be an exact JudgePrivacyPolicyV1 or None.")
    references: list[_TrustedPrivateJudgeReferenceBinding] = []
    for reference in private_references:
        if type(reference) is not _TrustedPrivateJudgeReferenceBinding:
            raise TypeError(
                "private_references must contain exact trusted private-reference bindings."
            )
        references.append(reference)
    if len({reference.key for reference in references}) != len(references):
        raise ValueError("Trusted private judge reference keys must be unique.")
    if price_book is not None and type(price_book) is not PriceBook:
        raise TypeError("price_book must be an exact PriceBook or None.")
    if candidate_route_relation not in {"independent_model", "same_model"}:
        raise ValueError("candidate_route_relation must identify same or independent routing.")
    implementation_revision = _model_judge_implementation_revision(
        key=validated_key,
        app=app,
        agent_name=validated_agent_name,
    )
    if profile is not None and profile.implementation_revision != implementation_revision:
        raise ValueError("Judge profile implementation does not match its trusted application.")
    if (profile is None) != (privacy_policy is None):
        raise ValueError("Judge profile and privacy policy must be supplied together.")
    if profile is not None and privacy_policy is not None:
        if profile.key != validated_key:
            raise ValueError("Judge profile key does not match its trusted binding.")
        if (profile.privacy_policy_key, profile.privacy_policy_revision) != (
            privacy_policy.key,
            privacy_policy.revision,
        ):
            raise ValueError("Judge profile privacy identity does not match its trusted policy.")
        expected_evidence = tuple(
            item
            for item, allowed in (
                ("final_output", True),
                ("transcript", privacy_policy.allow_transcript),
                ("public_reference", privacy_policy.allow_public_reference),
                ("private_reference", privacy_policy.allow_private_reference),
            )
            if allowed
        )
        if profile.allowed_evidence != expected_evidence:
            raise ValueError("Judge profile evidence permissions do not match its trusted policy.")
        if (profile.max_estimated_cost is None) != (price_book is None):
            raise ValueError("Judge profile cost identity does not match its trusted price book.")
        if price_book is not None and (
            profile.pricing_profile_fingerprint != pricing_profile_identity(price_book).fingerprint
        ):
            raise ValueError(
                "Judge profile pricing identity does not match its trusted price book."
            )
        if any(
            (reference.privacy_policy_key, reference.privacy_policy_revision)
            != (privacy_policy.key, privacy_policy.revision)
            for reference in references
        ):
            raise ValueError("Private judge references do not match the trusted privacy policy.")
    return _TrustedModelJudgeBinding(
        key=validated_key,
        app=app,
        agent_name=validated_agent_name,
        implementation_revision=implementation_revision,
        profile=(
            None
            if profile is None
            else JudgeProfileIdentityV1.model_validate(_model_python_input(profile))
        ),
        privacy_policy=(
            None
            if privacy_policy is None
            else JudgePrivacyPolicyV1.model_validate(_model_python_input(privacy_policy))
        ),
        private_references=tuple(references),
        price_book=None if price_book is None else copy_price_book(price_book),
        candidate_route_relation=candidate_route_relation,
        structured_app=(
            None if profile is None else _isolated_structured_judge_app(app, validated_agent_name)
        ),
    )


class _CompiledPortableAssertion(EvalAssertion):
    _app: CayuApp
    _assertion_revision: str
    _evidence_policy: EvaluationEvidencePolicySpec
    _pricing_binding: _CompiledPricingBinding
    _spec: AssertionSpec

    __slots__ = (
        "_app",
        "_assertion_revision",
        "_evidence_policy",
        "_pricing_binding",
        "_spec",
    )

    def __init__(
        self,
        spec: AssertionSpec,
        *,
        app: CayuApp,
        evidence_policy: EvaluationEvidencePolicySpec,
        pricing_binding: _CompiledPricingBinding,
    ) -> None:
        validated_spec = _validated_assertion_spec(spec)
        object.__setattr__(self, "_spec", validated_spec)
        object.__setattr__(self, "_assertion_revision", assertion_spec_revision(validated_spec))
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_evidence_policy", _validated_policy(evidence_policy))
        object.__setattr__(
            self,
            "_pricing_binding",
            pricing_binding
            if type(validated_spec) is MaxEstimatedCostAssertionSpec
            else _NO_PRICING,
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Compiled portable assertions are immutable.")

    @property
    def name(self) -> str:
        return self._spec.id

    @property
    def assertion_revision(self) -> str:
        return self._assertion_revision

    @property
    def evaluates_failed_session(self) -> bool:
        return type(self._spec) is RootStatusAssertionSpec

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        currencies = (
            (self._spec.currency,) if type(self._spec) is MaxEstimatedCostAssertionSpec else ()
        )
        pricing_snapshot = _pricing_snapshot_for_binding(self._pricing_binding)
        evidence = _build_assertion_evidence_view(
            context.trajectory,
            evidence_policy=self._evidence_policy,
            pricing_snapshot=pricing_snapshot,
            cost_currencies=currencies,
            app=self._app,
            root_evidence_available=context.root_evidence_available,
            allow_event_count_fallback=True,
            expected_pricing_profile_fingerprint=self._pricing_binding.fingerprint,
            bind_pricing_profile=True,
        )
        return self.evaluate_evidence(evidence)

    def evaluate_evidence(self, evidence: AssertionEvidenceView) -> EvalAssertionResult:
        if (
            type(self._spec) is MaxEstimatedCostAssertionSpec
            and evidence.pricing_profile_fingerprint != self._pricing_binding.fingerprint
        ):
            raise ValueError(
                "Assertion evidence pricing profile does not match the compiled contract."
            )
        return _evaluate_validated_assertion_spec(
            self._spec,
            evidence,
            known_revision=self._assertion_revision,
        )


class _CompiledModelJudgeAssertion(EvalAssertion):
    """One portable contract bound to a trusted, tool-free local judge."""

    _app: CayuApp
    _assertion_revision: str
    _binding: _TrustedModelJudgeBinding
    _evidence_policy: EvaluationEvidencePolicySpec
    _judge: LLMJudge
    _spec: ModelJudgeAssertionSpec

    __slots__ = (
        "_app",
        "_assertion_revision",
        "_binding",
        "_evidence_policy",
        "_judge",
        "_spec",
    )

    def __init__(
        self,
        spec: ModelJudgeAssertionSpec,
        *,
        binding: _TrustedModelJudgeBinding,
        app: CayuApp,
        evidence_policy: EvaluationEvidencePolicySpec,
    ) -> None:
        validated_spec = ModelJudgeAssertionSpec.model_validate(
            spec.model_dump(mode="python", round_trip=True, warnings="none")
        )
        if validated_spec.evaluator_key != binding.key:
            raise ValueError("Portable model judge key does not match its trusted evaluator.")
        object.__setattr__(self, "_spec", validated_spec)
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_evidence_policy", _validated_policy(evidence_policy))
        object.__setattr__(
            self,
            "_assertion_revision",
            assertion_spec_revision(validated_spec),
        )
        object.__setattr__(
            self,
            "_judge",
            LLMJudge(
                binding.app,
                agent_name=binding.agent_name,
                rubric=validated_spec.rubric,
                rubric_version=validated_spec.rubric_version,
                threshold=validated_spec.threshold,
                include_transcript=validated_spec.include_transcript,
                name=validated_spec.id,
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Compiled model judge assertions are immutable.")

    @property
    def name(self) -> str:
        return self._spec.id

    @property
    def assertion_revision(self) -> str:
        return self._assertion_revision

    def _with_resolved_implementation_revision(
        self,
        result: EvalAssertionResult,
    ) -> EvalAssertionResult:
        """Preserve the trusted binding for every published judge outcome."""

        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    _MODEL_JUDGE_RESOLVED_IMPLEMENTATION_REVISION_METADATA_KEY: (
                        self._binding.implementation_revision
                    ),
                }
            }
        )

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        evidence = _build_assertion_evidence_view(
            context.trajectory,
            evidence_policy=self._evidence_policy,
            pricing_snapshot=None,
            cost_currencies=(),
            app=self._app,
            root_evidence_available=context.root_evidence_available,
            allow_event_count_fallback=True,
            expected_pricing_profile_fingerprint=None,
            bind_pricing_profile=True,
        )
        return await self.evaluate_evidence(evidence, context)

    async def evaluate_evidence(
        self,
        evidence: AssertionEvidenceView,
        context: EvalContext,
    ) -> EvalAssertionResult:
        try:
            current_revision = _model_judge_implementation_revision(
                key=self._binding.key,
                app=self._binding.app,
                agent_name=self._binding.agent_name,
            )
        except Exception:
            return self._with_resolved_implementation_revision(
                self.error("Trusted model judge configuration became invalid.")
            )
        if current_revision != self._binding.implementation_revision:
            return self._with_resolved_implementation_revision(
                self.error("Trusted model judge implementation changed after compilation.")
            )
        if evidence.final_output_state != "complete":
            return self._with_resolved_implementation_revision(
                self.unavailable("Final-output evidence was not retained completely.")
            )
        task = _redacted_text(
            self._app,
            _first_user_text(context.transcript),
            "model judge task",
        )
        if len(task) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
            return self._with_resolved_implementation_revision(
                self.unavailable("Model-judge task evidence exceeded its portable bound.")
            )
        transcript = None
        if self._spec.include_transcript:
            transcript = _redacted_text(
                self._app,
                _render_transcript(context.transcript),
                "model judge transcript",
            )
            if len(transcript) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
                return self._with_resolved_implementation_revision(
                    self.unavailable("Model-judge transcript evidence exceeded its portable bound.")
                )
        result = await self._judge._evaluate_material(
            task=task,
            final_output=evidence.final_output,
            transcript_text=transcript,
        )
        return self._with_resolved_implementation_revision(
            EvalAssertionResult(
                name=self.name,
                assertion_revision=self.assertion_revision,
                outcome=result.outcome,
                score=result.score,
                threshold=result.threshold,
                message=result.message,
                metadata=result.metadata,
                cost_summary=result.cost_summary,
            )
        )


class _CompiledStructuredModelJudgeAssertion(EvalAssertion):
    """Structured corpus contract bound to one exact trusted judge profile."""

    _app: CayuApp
    _assertion_revision: str
    _binding: _TrustedModelJudgeBinding
    _evidence_policy: EvaluationEvidencePolicySpec
    _judge: StructuredLLMJudge
    _reference_identity: dict[str, object] | None
    _spec: StructuredModelJudgeAssertionSpec

    __slots__ = (
        "_app",
        "_assertion_revision",
        "_binding",
        "_evidence_policy",
        "_judge",
        "_reference_identity",
        "_spec",
    )

    def __init__(
        self,
        spec: StructuredModelJudgeAssertionSpec,
        *,
        binding: _TrustedModelJudgeBinding,
        app: CayuApp,
        evidence_policy: EvaluationEvidencePolicySpec,
    ) -> None:
        validated_spec = StructuredModelJudgeAssertionSpec.model_validate(_model_python_input(spec))
        profile = binding.profile
        policy = binding.privacy_policy
        if profile is None or policy is None or binding.structured_app is None:
            raise ValueError("Structured model judges require a complete trusted profile.")
        public_profile = profile.model_dump(mode="json")
        try:
            redacted_profile = app.redact_json(public_profile)
        except Exception as exc:
            raise ValueError(
                "Structured judge profile could not cross the candidate redaction boundary."
            ) from exc
        if redacted_profile != public_profile:
            raise ValueError("Structured judge profile contains a candidate workload secret.")
        if (validated_spec.judge_profile_key, validated_spec.judge_profile_revision) != (
            profile.key,
            profile.revision,
        ):
            raise ValueError("Structured model-judge profile does not match the trusted target.")
        if validated_spec.evidence.include_transcript and not policy.allow_transcript:
            raise ValueError("Judge profile does not permit transcript evidence.")
        reference = validated_spec.reference
        reference_text: str | None = None
        reference_identity: dict[str, object] | None = None
        if type(reference) is PublicJudgeReferenceV1:
            if not policy.allow_public_reference:
                raise ValueError("Judge profile does not permit public references.")
            public_reference = reference.model_dump(mode="json")
            try:
                redacted_reference = app.redact_json(public_reference)
            except Exception as exc:
                raise ValueError(
                    "Public judge reference could not cross the candidate redaction boundary."
                ) from exc
            if redacted_reference != public_reference:
                raise ValueError("Public judge reference contains a candidate workload secret.")
            reference_text = json.dumps(
                {
                    "expected_answer": reference.expected_answer,
                    "expected_facts": reference.expected_facts,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            reference_identity = {
                "kind": reference.kind,
                "id": reference.id,
                "revision": reference.revision,
            }
        elif type(reference) is PrivateJudgeReferenceV1:
            if not policy.allow_private_reference:
                raise ValueError("Judge profile does not permit private references.")
            if (
                reference.privacy_policy_key,
                reference.privacy_policy_revision,
            ) != (policy.key, policy.revision):
                raise ValueError("Private reference policy does not match the judge profile.")
            private = next(
                (item for item in binding.private_references if item.key == reference.key),
                None,
            )
            if private is None:
                raise ValueError(f"Private judge reference {reference.key!r} is unavailable.")
            if (
                private.revision,
                private.privacy_policy_key,
                private.privacy_policy_revision,
            ) != (
                reference.revision,
                reference.privacy_policy_key,
                reference.privacy_policy_revision,
            ):
                raise ValueError("Private judge reference revision or policy changed.")
            reference_text = private.content
            reference_identity = {
                "kind": reference.kind,
                "key": reference.key,
                "revision": reference.revision,
                "privacy_policy_key": reference.privacy_policy_key,
                "privacy_policy_revision": reference.privacy_policy_revision,
            }
        elif reference is not None:
            raise TypeError("Unsupported structured judge reference type.")
        if binding.candidate_route_relation == "same_model" and (
            profile.same_model_use != "allowed_and_labeled"
        ):
            raise ValueError(
                "Structured judge uses the candidate model but its profile forbids that route."
            )
        object.__setattr__(self, "_spec", validated_spec)
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_evidence_policy", _validated_policy(evidence_policy))
        object.__setattr__(self, "_reference_identity", reference_identity)
        object.__setattr__(
            self,
            "_assertion_revision",
            assertion_spec_revision(validated_spec),
        )
        object.__setattr__(
            self,
            "_judge",
            StructuredLLMJudge(
                binding.structured_app,
                judge_authority_app=binding.app,
                publication_app=app,
                agent_name=binding.agent_name,
                rubric=validated_spec.rubric,
                reference_text=reference_text,
                threshold=validated_spec.threshold,
                timeout_seconds=profile.timeout_seconds,
                max_input_tokens=profile.max_input_tokens,
                max_output_tokens=profile.max_output_tokens,
                max_total_tokens=profile.max_total_tokens,
                max_estimated_cost=profile.max_estimated_cost,
                cost_currency=profile.cost_currency or "USD",
                price_book=binding.price_book,
                publish_explanations=type(reference) is not PrivateJudgeReferenceV1,
                name=validated_spec.id,
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Compiled structured model judge assertions are immutable.")

    @property
    def name(self) -> str:
        return self._spec.id

    @property
    def assertion_revision(self) -> str:
        return self._assertion_revision

    def _with_public_contract(self, result: EvalAssertionResult) -> EvalAssertionResult:
        profile = self._binding.profile
        if profile is None:
            raise RuntimeError("Compiled structured judge lost its trusted profile.")
        raw_judgment = result.metadata.get(_STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY)
        judgment = dict(raw_judgment) if type(raw_judgment) is dict else {}
        public_record = {
            "judge_profile": profile.model_dump(mode="json"),
            "candidate_route_relation": self._binding.candidate_route_relation,
            "rubric_id": self._spec.rubric.id,
            "rubric_revision": self._spec.rubric.revision,
            "reference": self._reference_identity,
            **judgment,
        }
        return result.model_copy(
            update={
                "metadata": {
                    _STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY: public_record,
                }
            }
        )

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        evidence = _build_assertion_evidence_view(
            context.trajectory,
            evidence_policy=self._evidence_policy,
            pricing_snapshot=None,
            cost_currencies=(),
            app=self._app,
            root_evidence_available=context.root_evidence_available,
            allow_event_count_fallback=True,
            expected_pricing_profile_fingerprint=None,
            bind_pricing_profile=True,
        )
        if evidence.final_output_state != "complete":
            return self._with_public_contract(
                self.unavailable("Final-output evidence was not retained completely.")
            )
        task = _redacted_text(
            self._app,
            _first_user_text(context.transcript),
            "structured model judge task",
        )
        if len(task) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
            return self._with_public_contract(
                self.unavailable("Structured judge task evidence exceeded its portable bound.")
            )
        transcript = None
        if self._spec.evidence.include_transcript:
            transcript = _redacted_text(
                self._app,
                _render_transcript(context.transcript),
                "structured model judge transcript",
            )
            if len(transcript) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
                return self._with_public_contract(
                    self.unavailable(
                        "Structured judge transcript evidence exceeded its portable bound."
                    )
                )
        return await self.evaluate_retained_material(
            task=task,
            final_output=evidence.final_output,
            transcript_text=transcript,
        )

    async def evaluate_retained_material(
        self,
        *,
        task: str,
        final_output: str,
        transcript_text: str | None,
    ) -> EvalAssertionResult:
        """Judge one fixed evidence snapshot without candidate runtime execution."""

        if self._spec.evidence.include_transcript != (transcript_text is not None):
            return self._with_public_contract(
                self.unavailable("Fixed evidence does not match the transcript selection.")
            )
        try:
            current_revision = _model_judge_implementation_revision(
                key=self._binding.key,
                app=self._binding.app,
                agent_name=self._binding.agent_name,
            )
        except Exception:
            return self._with_public_contract(
                self.error("Trusted structured judge configuration became invalid.")
            )
        if current_revision != self._binding.implementation_revision:
            return self._with_public_contract(
                self.error("Trusted structured judge implementation changed after compilation.")
            )
        if len(task) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
            return self._with_public_contract(
                self.unavailable("Structured judge task evidence exceeded its portable bound.")
            )
        if len(final_output) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
            return self._with_public_contract(
                self.unavailable("Structured judge final-output evidence exceeded its bound.")
            )
        if (
            transcript_text is not None
            and len(transcript_text) > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS
        ):
            return self._with_public_contract(
                self.unavailable("Structured judge transcript evidence exceeded its bound.")
            )
        result = await self._judge._evaluate_material(
            task=task,
            final_output=final_output,
            transcript_text=transcript_text,
        )
        return self._with_public_contract(
            EvalAssertionResult(
                name=self.name,
                assertion_revision=self.assertion_revision,
                outcome=result.outcome,
                score=result.score,
                threshold=result.threshold,
                message=result.message,
                metadata=result.metadata,
                cost_summary=result.cost_summary,
            )
        )


def _prepare_portable_assertion_evidence(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
    *,
    runtime_app: CayuApp | None = None,
    memory_attribution_evidence: EvalMemoryAttributionEvidenceV1 | None = None,
) -> AssertionEvidenceView | None:
    trajectory = context.trajectory
    portable_assertions = tuple(
        assertion for assertion in assertions if type(assertion) is _CompiledPortableAssertion
    )
    model_judge_assertions = tuple(
        assertion for assertion in assertions if type(assertion) is _CompiledModelJudgeAssertion
    )
    structured_model_judge_assertions = tuple(
        assertion
        for assertion in assertions
        if type(assertion) is _CompiledStructuredModelJudgeAssertion
    )
    compiled = (
        *portable_assertions,
        *model_judge_assertions,
        *structured_model_judge_assertions,
    )
    if not compiled:
        return None

    first = compiled[0]
    if runtime_app is None:
        if any(assertion._app is not first._app for assertion in compiled[1:]):
            raise ValueError(
                "Compiled portable assertions must use one CayuApp redaction boundary."
            )
        projection_app = first._app
    else:
        if not isinstance(runtime_app, CayuApp):
            raise TypeError("runtime_app must be a CayuApp or None.")
        # The app executing a fresh trial owns its redaction boundary. The app
        # retained by a compiled assertion is used only when replay has no live
        # runtime app from which to derive that authority.
        projection_app = runtime_app
    if any(assertion._evidence_policy != first._evidence_policy for assertion in compiled[1:]):
        raise ValueError("Compiled portable assertions must use one evidence policy.")

    cost_assertions = tuple(
        assertion
        for assertion in portable_assertions
        if type(assertion._spec) is MaxEstimatedCostAssertionSpec
    )
    pricing_fingerprints = {assertion._pricing_binding.fingerprint for assertion in cost_assertions}
    if len(pricing_fingerprints) > 1:
        raise ValueError("Compiled cost assertions must use one canonical pricing profile.")
    pricing_fingerprint = next(iter(pricing_fingerprints), None)
    pricing_snapshot = _pricing_snapshot_for_bindings(
        tuple(assertion._pricing_binding for assertion in cost_assertions),
        expected_fingerprint=pricing_fingerprint,
    )
    currencies_set: set[str] = set()
    for assertion in cost_assertions:
        spec = assertion._spec
        if type(spec) is not MaxEstimatedCostAssertionSpec:
            raise AssertionError("Unreachable compiled cost assertion type.")
        currencies_set.add(spec.currency)
    currencies = _validated_currencies(tuple(sorted(currencies_set)))
    return _build_assertion_evidence_view(
        trajectory,
        evidence_policy=first._evidence_policy,
        pricing_snapshot=pricing_snapshot,
        cost_currencies=currencies,
        app=projection_app,
        root_evidence_available=context.root_evidence_available,
        allow_event_count_fallback=True,
        expected_pricing_profile_fingerprint=pricing_fingerprint,
        bind_pricing_profile=True,
        memory_attribution_evidence=memory_attribution_evidence,
    )


def _compiled_pricing_binding(pricing: PriceBook | None) -> _CompiledPricingBinding:
    if pricing is None:
        return _NO_PRICING
    return _CompiledPricingBinding(
        source=pricing,
        fingerprint=pricing_profile_identity(pricing).fingerprint,
    )


def _pricing_snapshot_for_binding(
    binding: _CompiledPricingBinding,
) -> _ValidatedPricingSnapshot | None:
    return _pricing_snapshot_for_bindings(
        (binding,),
        expected_fingerprint=binding.fingerprint,
    )


def _pricing_snapshot_for_bindings(
    bindings: Sequence[_CompiledPricingBinding],
    *,
    expected_fingerprint: str | None,
) -> _ValidatedPricingSnapshot | None:
    if expected_fingerprint is None:
        if any(binding.source is not None for binding in bindings):
            raise ValueError("Compiled pricing identity is inconsistent.")
        return None
    seen_sources: set[int] = set()
    selected_snapshot: _ValidatedPricingSnapshot | None = None
    for binding in bindings:
        source = binding.source
        if source is None:
            raise ValueError("Compiled pricing identity is inconsistent.")
        if id(source) in seen_sources:
            continue
        seen_sources.add(id(source))
        try:
            snapshot = _validated_pricing(source)
        except (TypeError, ValueError):
            raise ValueError(
                "Compiled pricing profile changed after assertion compilation."
            ) from None
        if snapshot is None or snapshot.identity.fingerprint != expected_fingerprint:
            raise ValueError("Compiled pricing profile changed after assertion compilation.")
        if selected_snapshot is None:
            selected_snapshot = snapshot
    if selected_snapshot is None:
        raise ValueError("Compiled pricing identity is inconsistent.")
    return selected_snapshot


def compile_assertion_spec(
    spec: AssertionSpec,
    *,
    app: CayuApp,
    evidence_policy: EvaluationEvidencePolicySpec,
    trusted_pricing: PriceBook | None,
) -> EvalAssertion:
    """Compile one authority-free spec into the existing EvalAssertion adapter."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    if trusted_pricing is not None and type(trusted_pricing) is not PriceBook:
        raise TypeError("trusted_pricing must be an exact PriceBook or None.")
    validated_spec = _validated_assertion_spec(spec)
    if type(validated_spec) in {
        ModelJudgeAssertionSpec,
        StructuredModelJudgeAssertionSpec,
    }:
        raise ValueError("Portable model judges require a trusted CorpusTarget evaluator binding.")
    pricing_binding = (
        _compiled_pricing_binding(trusted_pricing)
        if type(validated_spec) is MaxEstimatedCostAssertionSpec
        else _NO_PRICING
    )
    return _CompiledPortableAssertion(
        validated_spec,
        app=app,
        evidence_policy=evidence_policy,
        pricing_binding=pricing_binding,
    )


def _compile_corpus_assertion_specs(
    specs: Sequence[AssertionSpec],
    *,
    app: CayuApp,
    evidence_policy: EvaluationEvidencePolicySpec,
    trusted_pricing: PriceBook | None,
    expected_pricing_profile: PricingProfileIdentityV1 | None,
    trusted_pricing_identity: PricingProfileIdentityV1 | None = None,
    trusted_model_judges: Sequence[_TrustedModelJudgeBinding] = (),
) -> tuple[EvalAssertion, ...]:
    """Compile one corpus suite with a single shared pricing binding."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    if trusted_pricing is not None and type(trusted_pricing) is not PriceBook:
        raise TypeError("trusted_pricing must be an exact PriceBook or None.")
    if expected_pricing_profile is not None and (
        type(expected_pricing_profile) is not PricingProfileIdentityV1
    ):
        raise TypeError(
            "expected_pricing_profile must be an exact PricingProfileIdentityV1 or None."
        )
    if trusted_pricing_identity is not None and (
        type(trusted_pricing_identity) is not PricingProfileIdentityV1
    ):
        raise TypeError(
            "trusted_pricing_identity must be an exact PricingProfileIdentityV1 or None."
        )

    validated_specs = tuple(_validated_assertion_spec(spec) for spec in specs)
    model_judge_bindings: dict[str, _TrustedModelJudgeBinding] = {}
    for binding in trusted_model_judges:
        if type(binding) is not _TrustedModelJudgeBinding:
            raise TypeError("trusted_model_judges must contain exact trusted model judge bindings.")
        if binding.key in model_judge_bindings:
            raise ValueError("Trusted model judge keys must be unique.")
        model_judge_bindings[binding.key] = binding
    uses_pricing = any(type(spec) is MaxEstimatedCostAssertionSpec for spec in validated_specs)
    pricing_binding = _NO_PRICING
    if uses_pricing:
        if trusted_pricing is None:
            raise ValueError("Eval corpus pricing profile does not match the trusted CorpusTarget.")
        trusted_identity = (
            pricing_profile_identity(trusted_pricing)
            if trusted_pricing_identity is None
            else trusted_pricing_identity
        )
        if trusted_identity != expected_pricing_profile:
            raise ValueError("Eval corpus pricing profile does not match the trusted CorpusTarget.")
        pricing_binding = _CompiledPricingBinding(
            source=trusted_pricing,
            fingerprint=trusted_identity.fingerprint,
        )

    compiled: list[EvalAssertion] = []
    for spec in validated_specs:
        if type(spec) is ModelJudgeAssertionSpec:
            binding = model_judge_bindings.get(spec.evaluator_key)
            if binding is None:
                raise ValueError(
                    f"Eval corpus requires trusted model judge {spec.evaluator_key!r}."
                )
            compiled.append(
                _CompiledModelJudgeAssertion(
                    spec,
                    binding=binding,
                    app=app,
                    evidence_policy=evidence_policy,
                )
            )
        elif type(spec) is StructuredModelJudgeAssertionSpec:
            binding = model_judge_bindings.get(spec.judge_profile_key)
            if binding is None:
                raise ValueError(
                    f"Eval corpus requires trusted model judge {spec.judge_profile_key!r}."
                )
            compiled.append(
                _CompiledStructuredModelJudgeAssertion(
                    spec,
                    binding=binding,
                    app=app,
                    evidence_policy=evidence_policy,
                )
            )
        else:
            compiled.append(
                _CompiledPortableAssertion(
                    spec,
                    app=app,
                    evidence_policy=evidence_policy,
                    pricing_binding=pricing_binding,
                )
            )
    return tuple(compiled)
