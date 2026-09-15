"""Installed, credential-free benchmark acceptance material."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from cayu.evals.benchmark_package import (
    BenchmarkFileV1,
    BenchmarkPackageV1,
    BenchmarkRequirementsV1,
    benchmark_package_to_json,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    FinalOutputEqualsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
)
from cayu.evals.scenario import (
    EvalScenarioDocumentV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV3,
    EvalSuiteTrialRequestDraftV3,
    compile_eval_suite_draft_v3,
)

_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB1kAAAAASUVORK5CYII="
)


def synthetic_benchmark_package() -> BenchmarkPackageV1:
    """Return a versioned native suite including a real file-input scenario."""

    attachment = EvalScenarioDocumentV2.create(
        id="attachment",
        target_key="synthetic-benchmark",
        name="Native image input",
        events=(
            ScenarioInitialInputEventV2(
                id="input",
                sequence=0,
                input=ScenarioInputV2(
                    messages=(
                        ScenarioUserMessageV2(
                            content=(
                                ScenarioTextPartV2(text="Return exactly ATTACHED."),
                                ScenarioFilePartV2(artifact_requirement_id="pixel"),
                            )
                        ),
                    )
                ),
            ),
        ),
        artifact_requirements=(
            ScenarioArtifactRequirementV2(
                id="pixel",
                source="fixture_digest",
                content_sha256=hashlib.sha256(_PIXEL).hexdigest(),
                filename="pixel.png",
                content_type="image/png",
                size_bytes=len(_PIXEL),
            ),
        ),
    )
    cases = [
        EvalCaseDraftV2(
            id=case_id,
            name=case_id,
            stimulus=EvalSimpleInputStimulusV1(
                input=RunInputSpec(messages=(CorpusUserMessageSpec(text=prompt),))
            ),
            assertions=(
                RootStatusAssertionSpec(id="completed", expected="completed"),
                FinalOutputEqualsAssertionSpec(id="answer", expected=answer),
            ),
        )
        for case_id, prompt, answer in (
            ("echo", "Return exactly READY.", "READY"),
            ("wrong-answer", "Return exactly WRONG.", "CORRECT"),
        )
    ]
    cases.append(
        EvalCaseDraftV2(
            id="attachment",
            name="Native image input",
            stimulus=EvalScenarioStimulusV1(
                scenario_id=attachment.id,
                scenario_revision=attachment.revision,
            ),
            assertions=(
                RootStatusAssertionSpec(id="completed", expected="completed"),
                FinalOutputEqualsAssertionSpec(id="answer", expected="ATTACHED"),
            ),
        )
    )
    return BenchmarkPackageV1.create(
        id="synthetic",
        version="1.0.0",
        scorer_id="exact-answer",
        scorer_version="1",
        suite=compile_eval_suite_draft_v3(
            EvalSuiteDraftV3(
                id="synthetic-benchmark",
                target_key="synthetic-benchmark",
                name="Synthetic benchmark campaign",
                trial_request=EvalSuiteTrialRequestDraftV3(timeout_seconds=30),
                cases=tuple(cases),
            )
        ),
        scenarios=(attachment,),
        files=(
            BenchmarkFileV1(
                scenario_revision=attachment.revision,
                requirement_id="pixel",
                path="inputs/pixel.png",
            ),
        ),
        requirements=BenchmarkRequirementsV1(environments=("benchmark",)),
    )


def write_synthetic_benchmark_package(directory: str | Path) -> Path:
    """Create a new package directory; never overwrite existing user material."""

    destination = Path(directory)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    (destination / "inputs").mkdir(mode=0o700)
    (destination / "inputs" / "pixel.png").write_bytes(_PIXEL)
    manifest = destination / "benchmark.json"
    manifest.write_text(benchmark_package_to_json(synthetic_benchmark_package()), encoding="utf-8")
    return manifest
