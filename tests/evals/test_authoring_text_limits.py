from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.evals import corpus, suite_authoring
from cayu.evals import store as eval_store
from cayu.evals.corpus import (
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalCorpusSuiteInspectionV1,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
)
from cayu.evals.store import (
    EvalAuthoredSuiteCatalogEntry,
    EvalAuthoredSuiteCatalogQuery,
    EvalCaseCatalogEntry,
    EvalCaseCatalogQuery,
    EvalSuiteCatalogEntry,
    EvalSuiteCatalogQuery,
    InMemoryEvalStore,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalSuiteDraftV1,
    EvalSuiteDraftV2,
    EvalSuiteDraftV3,
    compile_eval_suite_draft_v3,
)
from cayu.server.contracts import (
    CapturedEvaluationCaseDraft,
    CapturedEvaluationSuiteDraft,
    EvaluationPromotionCaseDraft,
    EvaluationPromotionSuiteDraft,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.vaults.redaction import SecretRedactor

_CASES = json.loads(
    (Path(__file__).parents[2] / "dashboard/tests/fixtures/eval-text-boundaries.json").read_text()
)
_INPUT = {"messages": [{"role": "user", "text": "Run the example."}]}
_ASSERTIONS = [{"id": "completed", "kind": "root_status", "expected": "completed"}]
_CASE = {
    "id": "example",
    "name": "Example",
    "stimulus": {"kind": "simple_input", "input": _INPUT},
    "assertions": _ASSERTIONS,
}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
def test_eval_authoring_models_preserve_shared_text_boundaries(case):
    value = None if case["text"] is None else case["text"] * case["repeat"]
    metadata = {"name": "Example", "description": "Example description", case["field"]: value}
    material = {**_CASE, **metadata}
    source = EvaluationSourceIdentityV1(
        application_release_id="example-release",
        app_manifest_schema_version="7",
        app_manifest_fingerprint="a" * 64,
        evidence_revision="sha256:" + "b" * 64,
    )
    factories = {
        "corpus suite": lambda: EvalSuiteSpec.create(id="suite", **metadata),
        "corpus case": lambda: EvalCaseSpec.create(
            id="example",
            suite_id="suite",
            source=source,
            input=RunInputSpec.model_validate(_INPUT),
            assertions=(RootStatusAssertionSpec.model_validate(_ASSERTIONS[0]),),
            **metadata,
        ),
    }
    for model in (EvalCaseDraftV1, EvalCaseDraftV2):
        factories[model.__name__] = lambda model=model: model.model_validate(material)
    for model in (EvalSuiteDraftV1, EvalSuiteDraftV2, EvalSuiteDraftV3):
        factories[model.__name__] = lambda model=model: model.model_validate(
            {"id": "suite", "target_key": "target", "cases": [_CASE], **metadata}
        )
    catalog_fields = {"revision": "sha256:" + "a" * 64, "id": "example", **metadata}
    suite_counts = {"case_count": 1, "assertion_count": 1, "trials": 1, "timeout_seconds": 60}
    factories["suite catalog"] = lambda: EvalSuiteCatalogEntry(
        corpus_revision="sha256:" + "b" * 64, **catalog_fields, **suite_counts
    )
    factories["case catalog"] = lambda: EvalCaseCatalogEntry(
        corpus_revision="sha256:" + "b" * 64,
        suite_id="suite",
        message_count=1,
        assertion_count=1,
        **catalog_fields,
    )
    factories["authored suite catalog"] = lambda: EvalAuthoredSuiteCatalogEntry(
        suite_revision="sha256:" + "b" * 64,
        target_key="target",
        simple_input_count=1,
        scenario_count=0,
        document_bytes=1,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        **catalog_fields,
        **suite_counts,
    )
    if case["field"] == "name":
        factories["suite inspection"] = lambda: EvalCorpusSuiteInspectionV1(
            id="suite",
            revision="sha256:" + "a" * 64,
            name=value,
            case_count=1,
            assertion_count=1,
            trials=1,
            minimum_passed_trials=1,
            max_concurrency=1,
            timeout_seconds=60,
        )
    for name, create in factories.items():
        try:
            actual = create()
        except ValueError:
            accepted = False
        else:
            accepted = True
            assert getattr(actual, case["field"]) == value
        assert accepted is case["accepted"], (name, case["id"])


@pytest.mark.parametrize(("field", "maximum"), [("name", 256), ("description", 2048)])
def test_http_eval_drafts_preserve_scalar_length_limits(field, maximum):
    models = (
        (EvaluationPromotionSuiteDraft, {"trial_request": {}}),
        (CapturedEvaluationSuiteDraft, {}),
        (
            EvaluationPromotionCaseDraft,
            {"suite_id": "suite", "input": _INPUT, "assertions": _ASSERTIONS},
        ),
        (CapturedEvaluationCaseDraft, {"suite_id": "suite", "assertions": _ASSERTIONS}),
    )
    for model, extra in models:
        material = {"id": "example", "name": "Example", **extra, field: "😀" * maximum}
        assert getattr(model.model_validate(material), field) == material[field]
        schema = model.model_json_schema()["properties"][field]
        if field == "description":
            schema = schema["anyOf"][0]
        assert (schema["minLength"], schema["maxLength"]) == (1, maximum)
        with pytest.raises(ValidationError):
            model.model_validate({**material, field: "😀" * (maximum + 1)})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("field", "constant", "maximum"),
    [
        ("name", "EVAL_CORPUS_MAX_NAME_CHARS", 300),
        ("description", "EVAL_CORPUS_MAX_DESCRIPTION_CHARS", 4096),
    ],
    ids=["name", "description"],
)
def test_catalog_storage_follows_declared_text_limits(
    monkeypatch, tmp_path, backend, field, constant, maximum
):
    # Model a new declaration in its owner and the consumers' imported bindings.
    for module in (corpus, suite_authoring, eval_store):
        monkeypatch.setattr(module, constant, maximum)
    value = "😀" * maximum
    metadata = {"name": "Example", "description": "Description", field: value}
    authored = compile_eval_suite_draft_v3(
        EvalSuiteDraftV3.model_validate(
            {"id": "suite", "target_key": "target", "cases": [{**_CASE, **metadata}], **metadata}
        )
    )
    case = EvalCaseSpec.create(
        id="example",
        suite_id="suite",
        source=EvaluationSourceIdentityV1(
            application_release_id="example-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "b" * 64,
        ),
        input=RunInputSpec.model_validate(_INPUT),
        assertions=(RootStatusAssertionSpec.model_validate(_ASSERTIONS[0]),),
        **metadata,
    )
    document = EvalCorpusDocument.create(
        target_key="target",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(authored.suite,),
        cases=(case,),
    )

    async def exercise():
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "evals.db")
        )
        try:
            redact_json = SecretRedactor().redact_json
            await store.save_corpus(document, redact_json=redact_json)
            authored_entry = await store.save_authored_suite(authored, redact_json=redact_json)
            assert await store.load_corpus(document.revision) == document
            assert await store.load_authored_suite(authored.revision) == authored
            suites = await store.list_suites(
                EvalSuiteCatalogQuery(corpus_revision=document.revision)
            )
            cases = await store.list_cases(
                EvalCaseCatalogQuery(corpus_revision=document.revision, suite_id=authored.suite.id)
            )
            authored_page = await store.list_authored_suites(
                EvalAuthoredSuiteCatalogQuery(target_key="target")
            )
            assert len(suites.items) == len(cases.items) == 1
            assert authored_page.items == (authored_entry,)
            for entry in (*suites.items, *cases.items, *authored_page.items):
                assert getattr(entry, field) == value
                with pytest.raises(ValidationError, match=f"at most {maximum} characters"):
                    type(entry).model_validate(
                        {**entry.model_dump(mode="python"), field: value + "😀"}
                    )
        finally:
            await store.close()

    asyncio.run(exercise())
