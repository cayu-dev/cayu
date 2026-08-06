from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cayu.evals.assertions import EvalAssertion
from cayu.evals.corpus import (
    AssertionSpec,
    EvaluationEvidencePolicySpec,
    MaxEstimatedCostAssertionSpec,
    PricingProfileIdentityV1,
    RootStatusAssertionSpec,
    _validated_assertion_spec,
    assertion_spec_revision,
    pricing_profile_identity,
)
from cayu.evals.evidence import (
    AssertionEvidenceView,
    _build_assertion_evidence_view,
    _validated_currencies,
    _validated_policy,
    _validated_pricing,
    _ValidatedPricingSnapshot,
)
from cayu.evals.models import EvalAssertionResult, EvalContext
from cayu.evals.portable_evaluation import _evaluate_validated_assertion_spec
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook


@dataclass(frozen=True, slots=True)
class _CompiledPricingBinding:
    """Compile-time pricing identity backed by one shared trusted source."""

    source: PriceBook | None
    fingerprint: str | None


_NO_PRICING = _CompiledPricingBinding(source=None, fingerprint=None)


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


def _prepare_portable_assertion_evidence(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
    *,
    runtime_app: CayuApp | None = None,
) -> AssertionEvidenceView | None:
    trajectory = context.trajectory
    compiled = tuple(
        assertion for assertion in assertions if type(assertion) is _CompiledPortableAssertion
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
        for assertion in compiled
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

    validated_specs = tuple(_validated_assertion_spec(spec) for spec in specs)
    uses_pricing = any(type(spec) is MaxEstimatedCostAssertionSpec for spec in validated_specs)
    pricing_binding = _NO_PRICING
    if uses_pricing:
        if trusted_pricing is None:
            raise ValueError("Eval corpus pricing profile does not match the trusted CorpusTarget.")
        trusted_identity = pricing_profile_identity(trusted_pricing)
        if trusted_identity != expected_pricing_profile:
            raise ValueError("Eval corpus pricing profile does not match the trusted CorpusTarget.")
        pricing_binding = _CompiledPricingBinding(
            source=trusted_pricing,
            fingerprint=trusted_identity.fingerprint,
        )

    return tuple(
        _CompiledPortableAssertion(
            spec,
            app=app,
            evidence_policy=evidence_policy,
            pricing_binding=pricing_binding,
        )
        for spec in validated_specs
    )
