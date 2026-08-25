from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

import cayu
from cayu.knowledge_maintenance import (
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRoutedCandidate,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceRoutingResult,
    KnowledgeMaintenanceSignalKind,
    _candidate_payload_bytes,
)
from cayu.knowledge_maintenance_planning import (
    KnowledgeMaintenanceEvaluationFinding,
    KnowledgeMaintenanceEvaluationFindingCode,
    KnowledgeMaintenanceEvaluationFindingKind,
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenanceEvaluatorDecision,
    KnowledgeMaintenanceEvaluatorOutput,
    KnowledgeMaintenanceEvidenceMapping,
    KnowledgeMaintenanceInferenceUsage,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpoint,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlannerBudget,
    KnowledgeMaintenancePlannerOutput,
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningLimitExceeded,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningSnapshot,
    KnowledgeMaintenancePlanningWorkflow,
    KnowledgeMaintenanceRelationDraft,
    KnowledgeMaintenanceReplacementDraft,
)
from cayu.storage import (
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    MAX_KNOWLEDGE_REVISION,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
)

_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
_OLD = _NOW - timedelta(days=90)
_ACCESS = KnowledgeAccessScope.for_namespace(
    "project:cayu",
    required_labels={"project": "cayu"},
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE],
    include_expired=True,
)
_WRITE_ACCESS = KnowledgeAccessScope.privileged()


class _CountingStore(InMemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0
        self.fail_get = False
        self.block_get = False

    async def get_entry(self, *args: Any, **kwargs: Any):
        self.get_calls += 1
        if self.block_get:
            await asyncio.Event().wait()
        if self.fail_get:
            raise RuntimeError("private store failure")
        return await super().get_entry(*args, **kwargs)


class _Planner:
    def __init__(self, factory=None) -> None:
        self.factory = factory
        self.calls = 0
        self.inputs = []
        self.raise_error: BaseException | None = None
        self.raw_output: object | None = None
        self.usage = KnowledgeMaintenanceInferenceUsage()

    async def propose_maintenance(self, request):
        self.calls += 1
        self.inputs.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.raw_output is not None:
            return self.raw_output
        return KnowledgeMaintenancePlannerOutput(
            plan=(self.factory or _valid_plan)(request),
            usage=self.usage,
        )


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs = []
        self.raise_error: BaseException | None = None
        self.raw_output: object | None = None
        self.findings: tuple[KnowledgeMaintenanceEvaluationFinding, ...] = ()
        self.usage = KnowledgeMaintenanceInferenceUsage()
        self.before_return = None

    async def evaluate_maintenance_plan(self, request):
        self.calls += 1
        self.inputs.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.before_return is not None:
            await self.before_return()
        if self.raw_output is not None:
            return self.raw_output
        verdict = (
            KnowledgeMaintenanceEvaluationVerdict.REJECTED
            if self.findings
            else KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
        )
        return KnowledgeMaintenanceEvaluatorOutput(
            decision=KnowledgeMaintenanceEvaluatorDecision(
                plan_fingerprint=request.plan.fingerprint,
                routing_result_fingerprint=(
                    request.planner_input.snapshot.routing_result_fingerprint
                ),
                configuration_fingerprint=request.planner_input.configuration_fingerprint,
                verdict=verdict,
                findings=self.findings,
            ),
            usage=self.usage,
        )


class _NeverCalled:
    def __getattr__(self, name: str):
        async def fail(*_args, **_kwargs):
            raise AssertionError(f"{name} must not be called")

        return fail


def _entry(entry_id: str, *, revision: int = 1, text: str | None = None) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        revision=revision,
        text=text or f"Reviewed fact from {entry_id}.",
        namespace="project:cayu",
        labels={"project": "cayu"},
        visibility=KnowledgeVisibility.PROJECT,
        status=KnowledgeStatus.ACTIVE,
        created_at=_OLD,
        updated_at=_NOW if revision > 1 else _OLD,
        source_type="artifact",
        source_id=f"source:{entry_id}",
        source_hash=f"sha256:{entry_id}:{revision}",
    )


def _ref(entry_id: str, revision: int = 1) -> KnowledgeRevisionRef:
    return KnowledgeRevisionRef(entry_id=entry_id, revision=revision)


def _routing_request(*entry_ids: str) -> KnowledgeMaintenanceRoutingRequest:
    return KnowledgeMaintenanceRoutingRequest(
        id="maintenance-routing-1",
        policy_id="reviewed-consolidation-v1",
        namespace="project:cayu",
        labels={"project": "cayu"},
        access_scope=_ACCESS,
        signals=tuple(
            KnowledgeMaintenanceCandidateSignal(
                id=f"signal:{entry_id}",
                kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                references=(_ref(entry_id),),
                producer_id="test-suite",
                producer_version="1",
                reason_code="explicit_review",
                observed_at=_NOW,
            )
            for entry_id in entry_ids
        ),
        created_at=_NOW,
    )


async def _routed(store, *entry_ids: str, config=None):
    request = _routing_request(*entry_ids)
    result = await KnowledgeMaintenanceRouter(store, config=config).route(request)
    return request, result


def _config(**updates: Any) -> KnowledgeMaintenancePlanningConfig:
    planner_model_calls = updates.get("max_planner_model_calls", 1)
    evaluator_model_calls = updates.get("max_evaluator_model_calls", 1)
    updates.setdefault(
        "planner_model_ids",
        ("test-model",) if planner_model_calls > 0 else (),
    )
    updates.setdefault(
        "evaluator_model_ids",
        ("test-model",) if evaluator_model_calls > 0 else (),
    )
    return KnowledgeMaintenancePlanningConfig(
        planner_id="test-planner",
        planner_version="1",
        evaluator_id="test-evaluator",
        evaluator_version="1",
        **updates,
    )


def _snapshot(request, result) -> KnowledgeMaintenancePlanningSnapshot:
    return KnowledgeMaintenancePlanningSnapshot(
        request_id=request.id,
        policy_id=request.policy_id,
        namespace=request.namespace,
        labels=request.labels,
        routing_request_fingerprint=request.fingerprint,
        routing_result_fingerprint=result.fingerprint,
        routing_configuration_fingerprint=result.configuration_fingerprint,
        candidate_payload_bytes=result.candidate_payload_bytes,
        candidates=result.candidates,
        routed_signals=result.routed_signals,
    )


def _replacement_endpoint() -> KnowledgeMaintenancePlanEndpoint:
    return KnowledgeMaintenancePlanEndpoint(kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT)


def _source_endpoint(reference: KnowledgeRevisionRef) -> KnowledgeMaintenancePlanEndpoint:
    return KnowledgeMaintenancePlanEndpoint(
        kind=KnowledgeMaintenancePlanEndpointKind.SOURCE,
        reference=reference,
    )


def _valid_plan(request) -> KnowledgeMaintenancePlanDraft:
    sources = tuple(candidate.reference for candidate in request.snapshot.candidates)
    mappings = tuple(
        KnowledgeMaintenanceEvidenceMapping(
            id=f"claim:{index}",
            claim=f"Supported replacement claim {index}.",
            source_references=(reference,),
        )
        for index, reference in enumerate(sources)
    )
    relations = tuple(
        KnowledgeMaintenanceRelationDraft(
            id=f"relation:{index}",
            subject=_replacement_endpoint(),
            object=_source_endpoint(reference),
            kind=KnowledgeRelationKind.DERIVED_FROM,
            evidence_mapping_ids=(mappings[index].id,),
        )
        for index, reference in enumerate(sources)
    )
    return KnowledgeMaintenancePlanDraft(
        id="plan:1",
        routing_request_fingerprint=request.snapshot.routing_request_fingerprint,
        routing_result_fingerprint=request.snapshot.routing_result_fingerprint,
        configuration_fingerprint=request.configuration_fingerprint,
        policy_id=request.snapshot.policy_id,
        source_references=sources,
        replacement=KnowledgeMaintenanceReplacementDraft(
            text="A supported consolidation of the routed facts.",
            title="Consolidated fact",
            kind="fact",
            aspects=("maintenance",),
        ),
        relations=relations,
        evidence_mappings=mappings,
        rationale="The exact routed facts can be represented together.",
        evidence_summary="Every replacement claim maps to an exact source revision.",
    )


async def _prepared(*entry_ids: str):
    store = _CountingStore()
    for entry_id in entry_ids:
        await store.create_entry(_entry(entry_id), access_scope=_WRITE_ACCESS)
    request, result = await _routed(store, *entry_ids)
    store.get_calls = 0
    return store, request, result


def _run(coro):
    return asyncio.run(coro)


def test_public_planning_surface_is_exported() -> None:
    expected = {
        "KnowledgeMaintenanceEvaluationFindingCode",
        "KnowledgeMaintenancePlanDraft",
        "KnowledgeMaintenancePlanEvaluator",
        "KnowledgeMaintenancePlanner",
        "KnowledgeMaintenancePlannerBudget",
        "KnowledgeMaintenancePlanningConfig",
        "KnowledgeMaintenancePlanningSnapshot",
        "KnowledgeMaintenancePlanningWorkflow",
    }

    assert all(hasattr(cayu, name) for name in expected)


def test_plan_contract_is_canonical_strict_and_defensively_copied() -> None:
    async def run():
        _store, request, result = await _prepared("a", "b")
        planner_input = cayu.KnowledgeMaintenancePlannerInput(
            snapshot=_snapshot(request, result),
            configuration_fingerprint=_config().fingerprint,
            allowed_replacement_kinds=("fact",),
            budget=_config().planner_budget,
        )
        return _valid_plan(planner_input)

    plan = _run(run())
    reverse = plan.model_dump(mode="python")
    reverse["source_references"] = tuple(reversed(reverse["source_references"]))
    reverse["relations"] = tuple(reversed(reverse["relations"]))
    reverse["evidence_mappings"] = tuple(reversed(reverse["evidence_mappings"]))
    copied = KnowledgeMaintenancePlanDraft.model_validate(reverse)

    assert copied == plan
    assert copied.fingerprint == plan.fingerprint
    assert len(plan.fingerprint) == 64
    assert copied.replacement is not plan.replacement
    assert plan.replacement.aspects == ("maintenance",)

    with pytest.raises(ValidationError, match="extra"):
        KnowledgeMaintenancePlanDraft.model_validate(
            {**plan.model_dump(mode="python"), "write_authority": True}
        )
    with pytest.raises(ValidationError, match="replacement endpoint"):
        KnowledgeMaintenancePlanEndpoint(
            kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT,
            reference=_ref("a"),
        )
    with pytest.raises(ValidationError, match="one replacement and one source"):
        KnowledgeMaintenanceRelationDraft(
            id="bad",
            subject=_source_endpoint(_ref("a")),
            object=_source_endpoint(_ref("b")),
            kind=KnowledgeRelationKind.CONTRADICTS,
            evidence_mapping_ids=("claim:0",),
        )


def test_evaluation_diagnostics_use_only_closed_kind_bound_codes() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        KnowledgeMaintenanceEvaluationFinding.model_validate(
            {
                "kind": KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM,
                "code": "7365637265742d736f757263652d74657874",
            }
        )
    with pytest.raises(ValidationError, match="not valid for the selected finding kind"):
        KnowledgeMaintenanceEvaluationFinding(
            kind=KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM,
            code=KnowledgeMaintenanceEvaluationFindingCode.INFORMATION_LOSS,
        )
    with pytest.raises(ValidationError, match="extra"):
        KnowledgeMaintenanceEvaluatorDecision.model_validate(
            {
                "plan_fingerprint": "0" * 64,
                "routing_result_fingerprint": "1" * 64,
                "configuration_fingerprint": "2" * 64,
                "verdict": KnowledgeMaintenanceEvaluationVerdict.ACCEPTED,
                "code": "model_controlled_code",
            }
        )


def test_planning_snapshot_rejects_false_signal_and_payload_accounting() -> None:
    async def run():
        _store, request, result = await _prepared("a")
        return _snapshot(request, result)

    snapshot = _run(run())
    material = snapshot.model_dump(mode="python")
    material["candidate_payload_bytes"] += 1
    with pytest.raises(ValidationError, match="does not match the routed payload"):
        KnowledgeMaintenancePlanningSnapshot.model_validate(material)

    material = snapshot.model_dump(mode="python")
    material["candidates"][0]["signal_kinds"] = (KnowledgeMaintenanceSignalKind.EXPIRY,)
    with pytest.raises(ValidationError, match="signal kinds"):
        KnowledgeMaintenancePlanningSnapshot.model_validate(material)


def test_config_requires_separate_components_and_bounded_costs() -> None:
    component = _Planner()
    store = _CountingStore()

    with pytest.raises(ValueError, match="separate components"):
        KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=component,
            evaluator=component,
            config=_config(),
        )
    with pytest.raises(ValidationError, match="identities must be distinct"):
        KnowledgeMaintenancePlanningConfig(
            planner_id="same-component",
            planner_version="1",
            evaluator_id="same-component",
            evaluator_version="1",
        )
    with pytest.raises(ValidationError, match="Stage cost ceilings"):
        _config(
            max_planner_cost_micro_usd=60,
            max_evaluator_cost_micro_usd=60,
            max_total_cost_micro_usd=100,
        )
    with pytest.raises(ValidationError, match="positive planner model-call budget"):
        KnowledgeMaintenancePlanningConfig(
            planner_id="planner",
            planner_version="1",
            evaluator_id="evaluator",
            evaluator_version="1",
            evaluator_model_ids=("test-model",),
        )
    with pytest.raises(ValidationError, match="Zero planner model calls"):
        _config(
            max_planner_model_calls=0,
            planner_model_ids=("test-model",),
        )
    with pytest.raises(ValidationError, match="Zero model calls"):
        KnowledgeMaintenanceInferenceUsage(input_tokens=1)
    with pytest.raises(ValidationError, match="require `model_id`"):
        KnowledgeMaintenanceInferenceUsage(model_calls=1)
    with pytest.raises(ValidationError, match="at most 100"):
        _config(allowed_replacement_kinds=tuple(f"kind:{index}" for index in range(101)))


def test_zero_model_and_cost_budgets_support_deterministic_components() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        config = _config(
            max_planner_model_calls=0,
            max_evaluator_model_calls=0,
            max_planner_cost_micro_usd=0,
            max_evaluator_cost_micro_usd=0,
            max_total_cost_micro_usd=0,
        )
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=_Evaluator(),
            config=config,
            clock=lambda: _NOW,
        ).plan(request, routing)
        return config, result

    config, result = _run(run())

    assert config.planner_budget.max_model_calls == 0
    assert config.evaluator_budget.max_model_calls == 0
    assert result.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED
    assert result.planner_usage == KnowledgeMaintenanceInferenceUsage()
    assert result.evaluator_usage == KnowledgeMaintenanceInferenceUsage()


def test_workflow_accepts_only_after_three_currentness_checks_and_independent_evaluation() -> None:
    async def run():
        store, request, routing = await _prepared("a", "b")
        planner = _Planner()
        evaluator = _Evaluator()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        )
        result = await workflow.plan(request, routing)
        return store, planner, evaluator, result

    store, planner, evaluator, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED
    assert result.code == "evaluator_accepted"
    assert result.plan is not None
    assert result.evaluation is not None
    assert result.evaluation.evaluator_invoked
    assert result.evaluation.verdict is KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
    assert (result.planner_id, result.planner_version) == ("test-planner", "1")
    assert (result.evaluator_id, result.evaluator_version) == ("test-evaluator", "1")
    assert result.processed_at == _NOW
    assert result.planner_usage == KnowledgeMaintenanceInferenceUsage()
    assert result.evaluator_usage == KnowledgeMaintenanceInferenceUsage()
    assert store.get_calls == 6
    assert planner.calls == evaluator.calls == 1
    assert planner.inputs[0] is not evaluator.inputs[0].planner_input
    assert evaluator.inputs[0].plan is not result.plan
    assert isinstance(planner.inputs[0].budget, KnowledgeMaintenancePlannerBudget)
    assert planner.inputs[0].budget.max_evidence_mappings == 100
    assert planner.inputs[0].budget.max_replacement_text_bytes == 64 * 1024
    assert planner.inputs[0].budget.max_claim_bytes == cayu.MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES
    assert planner.inputs[0].budget.allowed_model_ids == ("test-model",)
    assert evaluator.inputs[0].budget.allowed_model_ids == ("test-model",)
    planner_payload = planner.inputs[0].model_dump_json()
    assert "access_scope" not in planner_payload
    assert "omissions" not in planner_payload
    assert "max_candidate_load_bytes" not in planner_payload
    assert len(result.fingerprint) == 64


def test_workflow_rejects_self_asserted_routing_result_bindings() -> None:
    async def run():
        store = _CountingStore()
        for entry_id in ("a", "b"):
            await store.create_entry(_entry(entry_id), access_scope=_WRITE_ACCESS)
        request, _routing_a = await _routed(store, "a")
        _request_b, routing_b = await _routed(store, "b")
        forged = routing_b.model_copy(
            update={
                "request_id": request.id,
                "request_fingerprint": request.fingerprint,
            }
        )
        store.get_calls = 0
        planner = _Planner()
        evaluator = _Evaluator()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
        )
        with pytest.raises(ValueError, match="does not bind"):
            await workflow.plan(request, forged)
        return store, planner, evaluator

    store, planner, evaluator = _run(run())

    assert store.get_calls == 0
    assert planner.calls == evaluator.calls == 0


def test_workflow_rejects_more_than_the_hard_source_limit_before_any_read() -> None:
    async def run():
        entry_ids = tuple(
            f"source-{index:02d}" for index in range(MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1)
        )
        store = _CountingStore()
        entries = {entry_id: _entry(entry_id) for entry_id in entry_ids}
        for entry in entries.values():
            await store.create_entry(entry, access_scope=_WRITE_ACCESS)
        request = _routing_request(*entry_ids)
        candidates = tuple(
            KnowledgeMaintenanceRoutedCandidate(
                reference=signal.references[0],
                entry=entries[signal.references[0].entry_id],
                signal_ids=(signal.id,),
                signal_kinds=(signal.kind,),
            )
            for signal in request.signals
        )
        payload_bytes = _candidate_payload_bytes(candidates, request.signals)
        forged_material = dict(
            schema_version=1,
            request_id=request.id,
            request_fingerprint=request.fingerprint,
            configuration_fingerprint="0" * 64,
            candidates=candidates,
            routed_signals=request.signals,
            omissions=(),
            signal_count=len(request.signals),
            loaded_reference_count=len(candidates),
            candidate_payload_bytes=payload_bytes,
            relation_payload_bytes=0,
            max_candidates=len(candidates),
            max_candidate_bytes=payload_bytes,
            max_relation_load_bytes=1,
            truncated=False,
        )
        with pytest.raises(ValidationError, match="max_candidates"):
            KnowledgeMaintenanceRoutingResult(**forged_material)
        forged = KnowledgeMaintenanceRoutingResult.model_construct(**forged_material)
        store.get_calls = 0
        planner = _Planner()
        evaluator = _Evaluator()
        with pytest.raises(KnowledgeMaintenancePlanningLimitExceeded) as exc_info:
            await KnowledgeMaintenancePlanningWorkflow(
                store,
                planner=planner,
                evaluator=evaluator,
                config=_config(),
            ).plan(request, forged)
        return store, planner, evaluator, exc_info.value

    store, planner, evaluator, error = _run(run())

    assert error.limit == "max_candidates"
    assert store.get_calls == planner.calls == evaluator.calls == 0


def test_zero_candidate_and_incomplete_routing_do_not_invoke_components_or_read_store() -> None:
    async def run():
        empty_store = _CountingStore()
        empty_request, empty_routing = await _routed(empty_store)
        empty_store.get_calls = 0
        planner = _Planner()
        evaluator = _Evaluator()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            empty_store,
            planner=planner,
            evaluator=evaluator,
            config=_config(max_planner_input_bytes=1),
            clock=lambda: _NOW,
        )
        empty = await workflow.plan(empty_request, empty_routing)

        limited_store = _CountingStore()
        for entry_id in ("a", "b"):
            await limited_store.create_entry(_entry(entry_id), access_scope=_WRITE_ACCESS)
        limited_request, limited_routing = await _routed(
            limited_store,
            "a",
            "b",
            config=KnowledgeMaintenanceRouterConfig(
                max_candidates=1,
                max_candidate_reads=2,
            ),
        )
        limited_store.get_calls = 0
        incomplete = await KnowledgeMaintenancePlanningWorkflow(
            limited_store,
            planner=planner,
            evaluator=evaluator,
            config=_config(max_planner_input_bytes=1),
            clock=lambda: _NOW,
        ).plan(limited_request, limited_routing)
        return empty_store, limited_store, planner, evaluator, empty, incomplete

    empty_store, limited_store, planner, evaluator, empty, incomplete = _run(run())

    assert empty.outcome is KnowledgeMaintenancePlanningOutcome.NO_CANDIDATES
    assert incomplete.outcome is KnowledgeMaintenancePlanningOutcome.ROUTING_INCOMPLETE
    assert empty_store.get_calls == limited_store.get_calls == 0
    assert planner.calls == evaluator.calls == 0


def test_planner_input_budget_fails_before_revalidation_or_invocation() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(max_planner_input_bytes=1),
        )
        with pytest.raises(
            KnowledgeMaintenancePlanningLimitExceeded,
            match="configured work limit",
        ) as exc_info:
            await workflow.plan(request, routing)
        return store, planner, evaluator, exc_info.value

    store, planner, evaluator, error = _run(run())

    assert error.limit == "max_planner_input_bytes"
    assert store.get_calls == planner.calls == evaluator.calls == 0


def test_revalidation_and_evaluator_input_limits_stop_before_the_bounded_stage() -> None:
    async def run():
        bounded_store, bounded_request, bounded_routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        with pytest.raises(KnowledgeMaintenancePlanningLimitExceeded) as revalidation_error:
            await KnowledgeMaintenancePlanningWorkflow(
                bounded_store,
                planner=planner,
                evaluator=evaluator,
                config=_config(max_revalidation_bytes=1),
            ).plan(bounded_request, bounded_routing)

        evaluator_store, evaluator_request, evaluator_routing = await _prepared("b")
        with pytest.raises(KnowledgeMaintenancePlanningLimitExceeded) as evaluator_error:
            await KnowledgeMaintenancePlanningWorkflow(
                evaluator_store,
                planner=planner,
                evaluator=evaluator,
                config=_config(max_evaluator_input_bytes=1),
            ).plan(evaluator_request, evaluator_routing)
        return (
            bounded_store,
            evaluator_store,
            planner,
            evaluator,
            revalidation_error.value,
            evaluator_error.value,
        )

    bounded_store, evaluator_store, planner, evaluator, revalidation_error, evaluator_error = _run(
        run()
    )

    assert revalidation_error.limit == "max_revalidation_bytes"
    assert evaluator_error.limit == "max_evaluator_input_bytes"
    assert bounded_store.get_calls == 0
    assert evaluator_store.get_calls == 2
    assert planner.calls == 1
    assert evaluator.calls == 0


@pytest.mark.parametrize(
    "config_updates",
    [
        {"max_plan_bytes": 1},
        {"max_evidence_mappings": 1},
        {"max_replacement_text_bytes": 1},
        {"max_claim_bytes": 1},
        {"max_planner_cost_micro_usd": 0},
    ],
)
def test_every_planner_output_ceiling_fails_before_evaluation(config_updates) -> None:
    async def run():
        store, request, routing = await _prepared("a", "b")
        planner = _Planner()
        if "max_planner_cost_micro_usd" in config_updates:
            planner.usage = KnowledgeMaintenanceInferenceUsage(
                model_calls=1,
                cost_micro_usd=1,
                model_id="test-model",
            )
        evaluator = _Evaluator()
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(**config_updates),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return store, evaluator, result

    store, evaluator, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
    assert result.planner_usage is not None
    assert store.get_calls == 2
    assert evaluator.calls == 0


@pytest.mark.parametrize("limit", ["bytes", "cost"])
def test_every_evaluator_output_ceiling_preserves_usage_and_stops_revalidation(limit) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        evaluator = _Evaluator()
        config_updates = {}
        if limit == "bytes":
            config_updates["max_evaluator_output_bytes"] = 1
        else:
            config_updates["max_evaluator_cost_micro_usd"] = 0
            evaluator.usage = KnowledgeMaintenanceInferenceUsage(
                model_calls=1,
                cost_micro_usd=1,
                model_id="test-model",
            )
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=evaluator,
            config=_config(**config_updates),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return store, result

    store, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET
    assert result.evaluator_usage is not None
    assert store.get_calls == 2


def test_source_revalidation_deadline_fails_closed_before_planning() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        store.block_get = True
        planner = _Planner()
        evaluator = _Evaluator()
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(source_revalidation_timeout_seconds=0.001),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return planner, evaluator, result

    planner, evaluator, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED
    assert planner.calls == evaluator.calls == 0


def test_changed_or_unreadable_sources_fail_closed_before_planning() -> None:
    async def run():
        stale_store, stale_request, stale_routing = await _prepared("a")
        await stale_store.append_entry_revision(
            _entry("a", revision=2, text="A concurrent update."),
            expected_revision=1,
            access_scope=_WRITE_ACCESS,
        )
        planner = _Planner()
        evaluator = _Evaluator()
        stale = await KnowledgeMaintenancePlanningWorkflow(
            stale_store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(stale_request, stale_routing)

        failed_store, failed_request, failed_routing = await _prepared("b")
        failed_store.fail_get = True
        failed = await KnowledgeMaintenancePlanningWorkflow(
            failed_store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(failed_request, failed_routing)
        return planner, evaluator, stale, failed

    planner, evaluator, stale, failed = _run(run())

    assert stale.outcome is KnowledgeMaintenancePlanningOutcome.SOURCE_SET_CHANGED
    assert failed.outcome is KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED
    assert planner.calls == evaluator.calls == 0
    assert "concurrent" not in stale.model_dump_json()
    assert "private" not in failed.model_dump_json()


@pytest.mark.parametrize(
    ("stage", "expected_outcome", "expected_reads"),
    [
        (
            "planner",
            KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_PLANNING,
            2,
        ),
        (
            "evaluator",
            KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_EVALUATION,
            3,
        ),
    ],
)
def test_post_component_revalidation_failures_remain_retryable(
    stage,
    expected_outcome,
    expected_reads,
) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        if stage == "planner":
            original = planner.propose_maintenance

            async def fail_after_planning(planner_input):
                output = await original(planner_input)
                store.fail_get = True
                return output

            planner.propose_maintenance = fail_after_planning  # type: ignore[invalid-assignment]
        else:

            async def fail_after_evaluation():
                store.fail_get = True

            evaluator.before_return = fail_after_evaluation
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return store, planner, evaluator, result

    store, planner, evaluator, result = _run(run())

    assert result.outcome is expected_outcome
    assert result.outcome is not KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED
    assert result.code == expected_outcome.value
    assert result.plan is not None
    assert result.evaluation is None
    assert result.planner_usage is not None
    assert (result.evaluator_usage is not None) is (stage == "evaluator")
    assert store.get_calls == expected_reads
    assert planner.calls == 1
    assert evaluator.calls == (stage == "evaluator")


@pytest.mark.parametrize(
    ("mutator", "expected_kind"),
    [
        (
            lambda plan: plan.model_copy(update={"policy_id": "different-policy"}),
            KnowledgeMaintenanceEvaluationFindingKind.ROUTING_BINDING_INVALID,
        ),
        (
            lambda plan: plan.model_copy(
                update={"source_references": (*plan.source_references, _ref("outside"))}
            ),
            KnowledgeMaintenanceEvaluationFindingKind.SOURCE_OUTSIDE_ROUTE,
        ),
        (
            lambda plan: plan.model_copy(update={"source_references": plan.source_references[:1]}),
            KnowledgeMaintenanceEvaluationFindingKind.SOURCE_COVERAGE_INCOMPLETE,
        ),
        (
            lambda plan: plan.model_copy(
                update={"replacement": plan.replacement.model_copy(update={"kind": "forbidden"})}
            ),
            KnowledgeMaintenanceEvaluationFindingKind.REPLACEMENT_KIND_DISALLOWED,
        ),
        (
            lambda plan: plan.model_copy(
                update={
                    "evidence_mappings": (
                        *plan.evidence_mappings,
                        KnowledgeMaintenanceEvidenceMapping(
                            id="claim:orphan",
                            claim="An unreferenced claim.",
                            source_references=(plan.source_references[0],),
                        ),
                    )
                }
            ),
            KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE,
        ),
    ],
)
def test_deterministic_evaluation_rejects_untrusted_plan_bindings(mutator, expected_kind) -> None:
    async def run():
        store, request, routing = await _prepared("a", "b")
        planner = _Planner(lambda planner_input: mutator(_valid_plan(planner_input)))
        evaluator = _Evaluator()
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return evaluator, result

    evaluator, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED
    assert result.evaluation is not None
    assert not result.evaluation.evaluator_invoked
    assert expected_kind in {finding.kind for finding in result.evaluation.findings}
    assert evaluator.calls == 0


def test_deterministic_evaluation_rejects_bad_relation_orientation_and_evidence() -> None:
    def invalid_plan(request):
        plan = _valid_plan(request)
        first = plan.source_references[0]
        bad_relation = plan.relations[0].model_copy(
            update={
                "subject": _source_endpoint(_ref(first.entry_id, MAX_KNOWLEDGE_REVISION)),
                "object": _replacement_endpoint(),
                "kind": KnowledgeRelationKind.SUPERSEDES,
                "evidence_mapping_ids": ("missing:mapping",),
            }
        )
        bad_mapping = plan.evidence_mappings[0].model_copy(
            update={"source_references": (_ref("outside"),)}
        )
        return plan.model_copy(
            update={
                "relations": (bad_relation, *plan.relations[1:]),
                "evidence_mappings": (bad_mapping, *plan.evidence_mappings[1:]),
            }
        )

    async def run():
        store, request, routing = await _prepared("a", "b")
        evaluator = _Evaluator()
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(invalid_plan),
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return evaluator, result

    evaluator, result = _run(run())
    assert result.outcome is KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED
    assert result.evaluation is not None
    kinds = {finding.kind for finding in result.evaluation.findings}
    assert KnowledgeMaintenanceEvaluationFindingKind.RELATION_ORIENTATION_INVALID in kinds
    assert KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_SOURCE_INVALID in kinds
    assert KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE in kinds
    assert KnowledgeMaintenanceEvaluationFindingKind.SOURCE_REVISION_EXHAUSTED in kinds
    assert evaluator.calls == 0


@pytest.mark.parametrize(
    "kind",
    [
        KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM,
        KnowledgeMaintenanceEvaluationFindingKind.INFORMATION_LOSS,
        KnowledgeMaintenanceEvaluationFindingKind.CONTRADICTION_MISHANDLED,
        KnowledgeMaintenanceEvaluationFindingKind.RETENTION_VIOLATION,
        KnowledgeMaintenanceEvaluationFindingKind.POLICY_VIOLATION,
        KnowledgeMaintenanceEvaluationFindingKind.PROMPT_INJECTION,
    ],
)
def test_independent_evaluator_can_reject_every_semantic_safety_class(kind) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        evaluator = _Evaluator()
        evaluator.findings = (
            KnowledgeMaintenanceEvaluationFinding(
                kind=kind,
                code=KnowledgeMaintenanceEvaluationFindingCode(kind.value),
                source_references=(_ref("a"),),
                evidence_mapping_ids=("claim:0",),
            ),
        )
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return result

    result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED
    assert result.code == "evaluator_rejected"
    assert result.evaluation is not None and result.evaluation.evaluator_invoked
    assert result.evaluation.code == "evaluator_rejected"
    assert result.evaluation.findings[0].kind is kind


@pytest.mark.parametrize(
    ("stage", "mode", "expected"),
    [
        ("planner", "raw", KnowledgeMaintenancePlanningOutcome.PLANNER_INVALID),
        ("planner", "error", KnowledgeMaintenancePlanningOutcome.PLANNER_FAILED),
        ("planner", "timeout", KnowledgeMaintenancePlanningOutcome.PLANNER_TIMED_OUT),
        ("planner", "budget", KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET),
        ("evaluator", "raw", KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID),
        ("evaluator", "error", KnowledgeMaintenancePlanningOutcome.EVALUATOR_FAILED),
        ("evaluator", "timeout", KnowledgeMaintenancePlanningOutcome.EVALUATOR_TIMED_OUT),
        ("evaluator", "budget", KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET),
    ],
)
def test_component_failures_are_safe_and_exhaustive(stage, mode, expected) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        component = planner if stage == "planner" else evaluator
        config_updates = {}
        if mode == "raw":
            component.raw_output = {"private": "content"}
        elif mode == "error":
            component.raise_error = RuntimeError("private component failure")
        elif mode == "timeout":

            async def block(*_args):
                await asyncio.Event().wait()

            if stage == "planner":
                planner.propose_maintenance = block
                config_updates["planner_timeout_seconds"] = 0.001
            else:
                evaluator.evaluate_maintenance_plan = block
                config_updates["evaluator_timeout_seconds"] = 0.001
        else:
            component.usage = KnowledgeMaintenanceInferenceUsage(
                model_calls=2,
                input_tokens=1,
                output_tokens=1,
                cost_micro_usd=1,
                model_id="test-model",
            )
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(**config_updates),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return result

    result = _run(run())

    assert result.outcome is expected
    assert "private" not in result.model_dump_json()
    if mode == "budget":
        usage = result.planner_usage if stage == "planner" else result.evaluator_usage
        assert usage is not None and usage.model_calls == 2


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("planner", KnowledgeMaintenancePlanningOutcome.PLANNER_INVALID),
        ("evaluator", KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID),
    ],
)
def test_component_cannot_reflect_an_unauthorized_model_identity(stage, expected) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        component = planner if stage == "planner" else evaluator
        component.usage = KnowledgeMaintenanceInferenceUsage(
            model_calls=1,
            input_tokens=1,
            output_tokens=1,
            cost_micro_usd=1,
            model_id="private source fragment",
        )
        return await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)

    result = _run(run())

    assert result.outcome is expected
    assert "private source fragment" not in result.model_dump_json()
    if stage == "planner":
        assert result.plan is None
        assert result.planner_usage is None
    else:
        assert result.plan is not None
        assert result.evaluator_usage is None


def test_authorized_model_usage_is_preserved_for_both_components() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        usage = KnowledgeMaintenanceInferenceUsage(
            model_calls=1,
            input_tokens=10,
            output_tokens=5,
            cost_micro_usd=100,
            model_id="test-model",
        )
        planner = _Planner()
        planner.usage = usage
        evaluator = _Evaluator()
        evaluator.usage = usage
        return usage, await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)

    usage, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED
    assert result.planner_usage == result.evaluator_usage == usage


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("planner", KnowledgeMaintenancePlanningOutcome.PLANNER_TIMED_OUT),
        ("evaluator", KnowledgeMaintenancePlanningOutcome.EVALUATOR_TIMED_OUT),
    ],
)
def test_component_cannot_turn_deadline_cancellation_into_success(stage, expected) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        component = planner if stage == "planner" else evaluator
        method_name = "propose_maintenance" if stage == "planner" else "evaluate_maintenance_plan"
        original = getattr(component, method_name)
        cancellation_seen = asyncio.Event()
        settled = asyncio.Event()

        async def suppress_cancellation(component_input):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await asyncio.sleep(0)
            try:
                return await original(component_input)
            finally:
                settled.set()

        setattr(component, method_name, suppress_cancellation)
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(**{f"{stage}_timeout_seconds": 0.001}),
            clock=lambda: _NOW,
        )
        result = await workflow.plan(request, routing)
        await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)
        await asyncio.wait_for(settled.wait(), timeout=1.0)
        return planner, evaluator, result

    planner, evaluator, result = _run(run())

    assert result.outcome is expected
    assert planner.calls == 1
    assert evaluator.calls == (stage == "evaluator")


def test_evaluator_cannot_switch_plan_or_forge_deterministic_diagnostics() -> None:
    async def run(mode: str):
        store, request, routing = await _prepared("a")
        evaluator = _Evaluator()

        async def invalid(request):
            finding_kind = (
                KnowledgeMaintenanceEvaluationFindingKind.SOURCE_OUTSIDE_ROUTE
                if mode == "kind"
                else KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM
            )
            finding = KnowledgeMaintenanceEvaluationFinding(
                kind=finding_kind,
                code=KnowledgeMaintenanceEvaluationFindingCode(finding_kind.value),
                source_references=(_ref("outside"),) if mode == "source" else (),
            )
            return KnowledgeMaintenanceEvaluatorOutput(
                decision=KnowledgeMaintenanceEvaluatorDecision(
                    plan_fingerprint=(
                        "0" * 64 if mode == "fingerprint" else request.plan.fingerprint
                    ),
                    routing_result_fingerprint=(
                        request.planner_input.snapshot.routing_result_fingerprint
                    ),
                    configuration_fingerprint=request.planner_input.configuration_fingerprint,
                    verdict=KnowledgeMaintenanceEvaluationVerdict.REJECTED,
                    findings=(finding,),
                )
            )

        evaluator.evaluate_maintenance_plan = invalid
        return await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)

    fingerprint = _run(run("fingerprint"))
    source = _run(run("source"))
    kind = _run(run("kind"))

    assert fingerprint.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
    assert source.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
    assert kind.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID


def test_source_revision_advancing_during_evaluation_invalidates_acceptance() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        evaluator = _Evaluator()

        async def advance():
            await store.append_entry_revision(
                _entry("a", revision=2, text="Changed during evaluation."),
                expected_revision=1,
                access_scope=_WRITE_ACCESS,
            )

        evaluator.before_return = advance
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return result

    result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED
    assert result.code == "source_set_changed_during_evaluation"
    assert result.evaluation is not None
    assert result.evaluation.verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED
    assert result.evaluation.findings[0].kind is (
        KnowledgeMaintenanceEvaluationFindingKind.STALE_SOURCE
    )


def test_source_revision_advancing_during_planning_skips_the_evaluator() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        original = planner.propose_maintenance

        async def advance_then_return(planner_input):
            output = await original(planner_input)
            await store.append_entry_revision(
                _entry("a", revision=2, text="Changed during planning."),
                expected_revision=1,
                access_scope=_WRITE_ACCESS,
            )
            return output

        planner.propose_maintenance = advance_then_return
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        return evaluator, result

    evaluator, result = _run(run())

    assert result.outcome is KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED
    assert result.code == "source_set_changed_during_planning"
    assert result.evaluation is not None and not result.evaluation.evaluator_invoked
    assert evaluator.calls == 0


@pytest.mark.parametrize("stage", ["source", "planner", "evaluator"])
def test_cancellation_propagates_without_becoming_a_failure(stage: str) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        if stage == "source":
            store.block_get = True
        elif stage == "planner":
            planner.raise_error = asyncio.CancelledError()
        else:
            evaluator.raise_error = asyncio.CancelledError()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
        )
        with pytest.raises(asyncio.CancelledError):
            if stage == "source":
                task = asyncio.create_task(workflow.plan(request, routing))
                await asyncio.sleep(0)
                task.cancel()
                await task
            else:
                await workflow.plan(request, routing)

    _run(run())


@pytest.mark.parametrize("stage", ["planner", "evaluator"])
def test_component_cannot_consume_caller_cancellation(stage: str) -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        component = planner if stage == "planner" else evaluator
        method_name = "propose_maintenance" if stage == "planner" else "evaluate_maintenance_plan"
        original = getattr(component, method_name)
        started = asyncio.Event()
        child_settled = asyncio.Event()

        async def suppress_cancellation(component_input):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
            try:
                return await original(component_input)
            finally:
                child_settled.set()

        setattr(component, method_name, suppress_cancellation)
        workflow = KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
        )
        planning = asyncio.create_task(workflow.plan(request, routing))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        planning.cancel()
        with pytest.raises(asyncio.CancelledError):
            await planning
        await asyncio.wait_for(child_settled.wait(), timeout=1.0)

    _run(run())


def test_component_inputs_are_copies_and_cannot_mutate_routing_or_plan() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        planner = _Planner()
        evaluator = _Evaluator()
        result = await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=planner,
            evaluator=evaluator,
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)
        planner.inputs[0].snapshot.candidates[0].entry.labels["outside"] = "planner"
        return request, routing, result

    request, routing, result = _run(run())

    assert "outside" not in routing.candidates[0].entry.labels
    assert "outside" not in request.labels
    assert result.plan is not None
    assert result.plan.replacement.aspects == ("maintenance",)


def test_result_contract_rejects_incomplete_or_mismatched_acceptance_proof() -> None:
    async def run():
        store, request, routing = await _prepared("a")
        return await KnowledgeMaintenancePlanningWorkflow(
            store,
            planner=_Planner(),
            evaluator=_Evaluator(),
            config=_config(),
            clock=lambda: _NOW,
        ).plan(request, routing)

    accepted = _run(run())
    assert accepted.evaluation is not None
    rejected = accepted.model_dump(mode="python")
    rejected["evaluation"]["verdict"] = KnowledgeMaintenanceEvaluationVerdict.REJECTED
    rejected["evaluation"]["findings"] = (
        KnowledgeMaintenanceEvaluationFinding(
            kind=KnowledgeMaintenanceEvaluationFindingKind.POLICY_VIOLATION,
            code=KnowledgeMaintenanceEvaluationFindingCode.POLICY_VIOLATION,
        ),
    )
    with pytest.raises(ValidationError, match="accepted result"):
        type(accepted).model_validate(rejected)

    not_invoked = accepted.model_dump(mode="python")
    not_invoked["evaluation"]["evaluator_invoked"] = False
    with pytest.raises(ValidationError, match="accepted result"):
        type(accepted).model_validate(not_invoked)

    missing_usage = accepted.model_dump(mode="python")
    missing_usage["evaluator_usage"] = None
    with pytest.raises(ValidationError, match="Evaluator usage"):
        type(accepted).model_validate(missing_usage)

    mismatched_plan = accepted.model_dump(mode="python")
    mismatched_plan["plan"]["routing_request_fingerprint"] = "0" * 64
    with pytest.raises(ValidationError, match="plan does not bind"):
        type(accepted).model_validate(mismatched_plan)

    rejected_by_evaluator = accepted.model_copy(
        update={
            "outcome": KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED,
            "code": "rejected",
            "evaluation": accepted.evaluation.model_copy(
                update={
                    "verdict": KnowledgeMaintenanceEvaluationVerdict.REJECTED,
                    "code": "rejected",
                    "findings": (
                        KnowledgeMaintenanceEvaluationFinding(
                            kind=KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM,
                            code=KnowledgeMaintenanceEvaluationFindingCode.UNSUPPORTED_CLAIM,
                            source_references=(_ref("outside"),),
                        ),
                    ),
                }
            ),
        }
    )
    with pytest.raises(ValidationError, match="findings do not bind"):
        type(accepted).model_validate(rejected_by_evaluator.model_dump(mode="python"))
