"""Installed, credential-free benchmark acceptance material."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
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
from cayu.evals.execution import CorpusExecutionLimits, CorpusTarget
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.runner import EvalPlan
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
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import FilePart, TextPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore

_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB1kAAAAASUVORK5CYII="
)


def synthetic_benchmark_package(
    *, failure_modes: bool = False, interruption_case: bool = False, scorer_version: str = "1"
) -> BenchmarkPackageV1:
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
    if interruption_case:
        cases.append(
            EvalCaseDraftV2(
                id="interruption",
                name="Pending synthetic execution",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Hold synthetic execution."),)
                    )
                ),
                assertions=(FinalOutputEqualsAssertionSpec(id="answer", expected="READY"),),
            )
        )
    if failure_modes:
        from cayu.evals.corpus import ModelJudgeAssertionSpec

        cases.extend(
            (
                EvalCaseDraftV2(
                    id="provider-failure",
                    name="Provider failure followed by clean retry",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(
                            messages=(CorpusUserMessageSpec(text="Fail once, then return READY."),)
                        )
                    ),
                    assertions=(FinalOutputEqualsAssertionSpec(id="answer", expected="READY"),),
                ),
                EvalCaseDraftV2(
                    id="scoring-failure",
                    name="Malformed synthetic model judgment",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(
                            messages=(CorpusUserMessageSpec(text="Return exactly READY."),)
                        )
                    ),
                    assertions=(
                        ModelJudgeAssertionSpec(
                            id="judge",
                            evaluator_key="synthetic-judge",
                            rubric="Accept READY.",
                            rubric_version="1",
                        ),
                    ),
                ),
            )
        )
    return BenchmarkPackageV1.create(
        id="synthetic",
        version="1.0.0",
        scorer_id="exact-answer",
        scorer_version=scorer_version,
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


def write_synthetic_benchmark_package(
    directory: str | Path,
    *,
    failure_modes: bool = False,
    interruption_case: bool = False,
    scorer_version: str = "1",
) -> Path:
    """Create a new package directory; never overwrite existing user material."""

    destination = Path(directory)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    (destination / "inputs").mkdir(mode=0o700)
    (destination / "inputs" / "pixel.png").write_bytes(_PIXEL)
    manifest = destination / "benchmark.json"
    manifest.write_text(
        benchmark_package_to_json(
            synthetic_benchmark_package(
                failure_modes=failure_modes,
                interruption_case=interruption_case,
                scorer_version=scorer_version,
            )
        ),
        encoding="utf-8",
    )
    return manifest


def _synthetic_response(request: ModelRequest):
    text = " ".join(
        part.text
        for message in request.messages
        if message.role == "user"
        for part in message.content
        if type(part) is TextPart
    )
    if any(type(part) is FilePart for message in request.messages for part in message.content):
        if not request.options.get("cayu_file_attachments"):
            raise RuntimeError("Synthetic native attachment was not resolved.")
        answer = "ATTACHED"
    elif "Return exactly WRONG." in text:
        answer = "WRONG"
    else:
        answer = "READY"
    return (
        ModelStreamEvent.text_delta(answer),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
            }
        ),
    )


class _SyntheticBenchmarkProvider(ScriptedModelProvider):
    def __init__(self, root: Path):
        self._benchmark_root = root
        super().__init__(response_factory=self._respond)

    def _respond(self, request: ModelRequest):
        text = " ".join(
            part.text
            for message in request.messages
            for part in message.content
            if type(part) is TextPart
        )
        # This local fixture journal is an acceptance counter, never provider billing.
        with (self._benchmark_root / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "model": request.model,
                        "kind": "judge" if request.model == "synthetic-judge" else "candidate",
                    }
                )
                + "\n"
            )
        if "Fail once, then return READY." in text:
            marker = self._benchmark_root / "provider-failure-once"
            try:
                marker.touch(exist_ok=False)
            except FileExistsError:
                pass
            else:
                raise RuntimeError("Synthetic provider failure.")
        return _synthetic_response(request)

    async def stream(self, request: ModelRequest):
        events = self._consume_batch(request)
        text = " ".join(
            part.text
            for message in request.messages
            for part in message.content
            if type(part) is TextPart
        )
        if "Hold synthetic execution." in text:
            await asyncio.sleep(90)
        for event in events:
            yield event

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="cayu-synthetic-benchmark",
            behavior_version="2",
            implementation_version="1",
        )


def build_synthetic_benchmark_plan(
    storage_directory: str | Path = ".cayu/synthetic-benchmark",
) -> EvalPlan:
    """Native durable target for installed-wheel acceptance; makes no external calls."""

    root = Path(storage_directory).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    app = CayuApp(session_store=SQLiteSessionStore(root / "sessions.sqlite3"), enable_logging=False)
    app.register_provider(_SyntheticBenchmarkProvider(root), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="benchmark",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="synthetic-benchmark-environment",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            artifact_store=LocalArtifactStore(
                root / "artifacts", store_id="synthetic-benchmark-v1"
            ),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="synthetic", model="synthetic-v1"))
    app.register_agent(AgentSpec(name="synthetic-judge", model="synthetic-judge"))
    from cayu.evals.execution import ModelJudgeTarget

    return EvalPlan(
        corpus_target=CorpusTarget(
            key="synthetic-benchmark",
            app=app,
            request_base=RunRequest(
                agent_name="synthetic", messages=[], environment_name="benchmark", max_steps=1
            ),
            application_release_id="synthetic-benchmark-v1",
            limits=CorpusExecutionLimits(max_trials=100, max_concurrency=32),
            model_judges=(
                ModelJudgeTarget(key="synthetic-judge", app=app, agent_name="synthetic-judge"),
            ),
        ),
        execution_profile_policy=EvalExecutionProfilePolicyV1(
            fixture_strategy="application_managed",
            reset_strategy="application_managed",
            effect_posture="isolated_application_authority",
            isolation_revision="sha256:"
            + hashlib.sha256(b"synthetic-no-external-effects-v1").hexdigest(),
            max_trials=100,
            max_concurrency=32,
        ),
    )
