"""Compile a reviewed session addition from current server-owned evidence."""

from __future__ import annotations

from fastapi import HTTPException

from cayu.evals.corpus import (
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    StructuredModelJudgeAssertionSpec,
    _content_revision,
)
from cayu.evals.promotion import (
    CapturedEvaluationCandidateV1,
    build_promotion_candidate,
    corpus_from_captured_evaluation_candidate,
)
from cayu.evals.read_views import EvalSuiteViewIdentity, decode_suite_view
from cayu.evals.results import CapturedEvaluationResultV1, EvalResultTargetIdentityV1
from cayu.evals.session_additions import (
    EvalAdditionPlan,
    EvalAdditionRequest,
    EvalExistingDestination,
    PreparedEvalAddition,
    addition_revision,
    addition_source_binding,
)
from cayu.evals.store import EvalStore
from cayu.evals.suite_authoring import (
    EvalCaseDefinitionV1,
    EvalCaseDefinitionV2,
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV1,
    EvalSuiteDraftV3,
    StructuredModelJudgeAssertionDraftV1,
    compile_eval_suite_authoring_draft,
)
from cayu.server._captured_evaluation_source import CapturedEvaluationSource
from cayu.server._evaluation_promotion_routes import _require_safe_promotion_document


def _append_case(cases, case):
    existing = next((item for item in cases if item.id == case.id), None)
    if existing is not None:
        if existing != case:
            raise HTTPException(
                status_code=409, detail="The destination already has a different case with this id."
            )
        return cases
    return tuple(sorted((*cases, case), key=lambda item: item.id))


class EvalAdditionPlanner:
    def __init__(self, source: CapturedEvaluationSource, store: EvalStore):
        self.source = source
        self.store = store

    async def prepare(self, session_id: str, request: EvalAdditionRequest) -> PreparedEvalAddition:
        _require_safe_promotion_document(
            self.source.app,
            request.model_dump(mode="json"),
            code="draft_rejected",
            failure_subject="The session addition",
        )
        trajectory, registration, baseline = await self.source.load(session_id)
        assert trajectory.session is not None
        parent = None
        parent_document = None
        destination = request.destination
        if isinstance(destination, EvalExistingDestination):
            parent = decode_suite_view(destination.selector)
            if not isinstance(parent, EvalSuiteViewIdentity):
                raise HTTPException(
                    status_code=422, detail="Reported series cannot receive session additions."
                )
            if parent.target_key != baseline.target_key:
                raise HTTPException(
                    status_code=409, detail="Session and destination targets differ."
                )
            if parent.source == "authored":
                parent_document = await self.store.load_authored_suite(parent.revision)
                suite = None if parent_document is None else parent_document.suite
            else:
                parent_document = await self.store.load_corpus(parent.revision)
                suite = (
                    None
                    if parent_document is None
                    else next(
                        (item for item in parent_document.suites if item.id == parent.suite_id),
                        None,
                    )
                )
            if (
                parent_document is None
                or suite is None
                or parent_document.target_key != baseline.target_key
            ):
                raise HTTPException(status_code=404, detail="Destination suite not found.")
        else:
            suite = EvalSuiteSpec.create(id=destination.suite_id, name=destination.name)

        assertions = baseline.case.assertions if request.assertions is None else request.assertions
        captured = None
        authored = None
        corpus = None
        if request.mode == "score":
            # A captured result keeps an independent one-case execution contract.
            capture_suite = (
                suite
                if parent is None
                else EvalSuiteSpec.create(id=baseline.suite.id, name=baseline.suite.name)
            )
            case = EvalCaseSpec.create(
                id=request.case_id,
                suite_id=capture_suite.id,
                name=request.case_name,
                source=baseline.source.case_source(),
                input=None,
                assertions=assertions,
            )
            candidate = CapturedEvaluationCandidateV1.create(
                target_key=baseline.target_key,
                source=baseline.source,
                evidence_policy=baseline.evidence_policy,
                evidence=baseline.evidence,
                pricing_profile=baseline.pricing_profile,
                suite=capture_suite,
                case=case,
            )
            score = self.source.score(trajectory, registration, candidate)
            corpus = corpus_from_captured_evaluation_candidate(candidate)
            captured = CapturedEvaluationResultV1.create(
                corpus=corpus,
                target=EvalResultTargetIdentityV1(
                    target_key=baseline.target_key,
                    application_release_id=baseline.source.application_release_id,
                    app_manifest_schema_version=baseline.source.app_manifest_schema_version,
                    app_manifest_fingerprint=baseline.source.app_manifest_fingerprint,
                ),
                score=score,
            )
            resulting = parent or EvalSuiteViewIdentity(
                source="corpus",
                target_key=corpus.target_key,
                revision=corpus.revision,
                suite_id=suite.id,
            )
            membership_changed = parent is None
        else:
            target = registration.target
            runnable = build_promotion_candidate(
                self.source.app,
                trajectory,
                target_key=target.key,
                source_agent_name=target.request_base.agent_name,
                application_release_id=target.application_release_id,
                evidence_policy=target.evidence_policy,
                pricing=target.price_book,
                project_root=registration.manifest_project_root,
            )
            if runnable.case.input is None:
                raise HTTPException(
                    status_code=409, detail="The captured session has no runnable input."
                )
            if isinstance(parent_document, EvalCorpusDocument):
                case = EvalCaseSpec.create(
                    id=request.case_id,
                    suite_id=suite.id,
                    name=request.case_name,
                    source=runnable.source.case_source(),
                    input=runnable.case.input,
                    assertions=assertions,
                )
                corpus = EvalCorpusDocument.create(
                    target_key=parent_document.target_key,
                    evidence_policy=parent_document.evidence_policy,
                    pricing_profile=parent_document.pricing_profile,
                    suites=parent_document.suites,
                    cases=_append_case(parent_document.cases, case),
                )
                resulting = EvalSuiteViewIdentity(
                    source="corpus",
                    target_key=corpus.target_key,
                    revision=corpus.revision,
                    suite_id=suite.id,
                )
            else:
                draft_assertions = (
                    assertions
                    if type(parent_document) is EvalSuiteDocumentV1
                    else tuple(
                        StructuredModelJudgeAssertionDraftV1.from_assertion(assertion)
                        if type(assertion) is StructuredModelJudgeAssertionSpec
                        else assertion
                        for assertion in assertions
                    )
                )
                draft_material = dict(
                    id=request.case_id,
                    name=request.case_name,
                    source=runnable.source.case_source(),
                    stimulus=EvalSimpleInputStimulusV1(input=runnable.case.input),
                    assertions=draft_assertions,
                )
                if parent_document is None:
                    authored = compile_eval_suite_authoring_draft(
                        EvalSuiteDraftV3(
                            id=suite.id,
                            name=suite.name,
                            target_key=baseline.target_key,
                            cases=(EvalCaseDraftV2.model_validate(draft_material),),
                        )
                    )
                    case = authored.cases[0]
                else:
                    case = (
                        EvalCaseDefinitionV1.create(EvalCaseDraftV1.model_validate(draft_material))
                        if type(parent_document) is EvalSuiteDocumentV1
                        else EvalCaseDefinitionV2.create(
                            EvalCaseDraftV2.model_validate(draft_material)
                        )
                    )
                    material = parent_document.model_dump(
                        mode="json", exclude={"revision", "cases"}
                    )
                    material["cases"] = [
                        item.model_dump(mode="json")
                        for item in _append_case(parent_document.cases, case)
                    ]
                    authored = type(parent_document)(
                        revision=_content_revision(material, "authored eval suite"),
                        **material,
                    )
                resulting = EvalSuiteViewIdentity(
                    source="authored",
                    target_key=authored.target_key,
                    revision=authored.revision,
                )
            membership_changed = resulting != parent
        destination_document = (
            parent_document
            if request.mode == "score" and parent is not None
            else authored or corpus
        )
        assert destination_document is not None
        destination_cases = (
            tuple(
                item for item in destination_document.cases if item.suite_id == resulting.suite_id
            )
            if isinstance(destination_document, EvalCorpusDocument)
            else destination_document.cases
        )
        material = dict(
            mode=request.mode,
            source_revision=baseline.revision,
            source_binding_revision=addition_source_binding(
                session_id, trajectory.session.id, baseline.revision
            ),
            parent=None if parent is None else parent.model_dump(mode="json"),
            destination=resulting.model_dump(mode="json"),
            case_id=case.id,
            case_revision=case.revision,
            case_name=case.name,
            case_definition=case.model_dump(mode="json"),
            destination_case_count=len(destination_cases),
            membership_changed=membership_changed,
            execution_case_count=1 if captured is not None else None,
            captured_result_revision=None if captured is None else captured.revision,
        )
        plan = EvalAdditionPlan.model_validate(
            {"revision": addition_revision(material), **material}
        )
        return PreparedEvalAddition(
            plan=plan,
            source=baseline,
            private_session_id=trajectory.session.id,
            authored=authored,
            corpus=corpus,
            captured=captured,
        )
