"""Portable distribution of existing authored suites and scenario fixtures.

A package is data, not execution authority. Loading it never imports an application,
creates an environment, or runs an evaluator. Only named scenario input files are
materialized at launch; the suite and its reference assertions stay evaluator-side.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
)
from cayu.evals.corpus import (
    EvaluationSourceIdentityV1,
    _bounded_durable_text,
    _content_revision,
    _model_content_revision,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
)
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.suite_authoring import (
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSuiteDocumentV3,
    EvalSuiteDraftV3,
    EvalSuiteSelectionV1,
    compile_eval_suite_draft_v3,
    eval_suite_selection,
)

BENCHMARK_PACKAGE_MAX_BYTES = 8 * 1024 * 1024
BENCHMARK_FILE_MAX_BYTES = 16 * 1024 * 1024
BENCHMARK_FILES_MAX_BYTES = 64 * 1024 * 1024


class BenchmarkFileV1(_PortableModel):
    """A package-relative source for one existing immutable scenario requirement."""

    scenario_revision: StrictStr
    requirement_id: StrictStr
    path: StrictStr

    @field_validator("scenario_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("requirement_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _bounded_durable_text(value, "path", max_chars=512, nonblank=True, clean=True)
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or path.as_posix() != value
        ):
            raise ValueError("Benchmark files require normalized relative POSIX paths.")
        return value


class BenchmarkRequirementsV1(_PortableModel):
    """Names which must already exist on the explicitly selected trusted target."""

    environments: tuple[StrictStr, ...] = Field(default=(), max_length=64)
    tools: tuple[StrictStr, ...] = Field(default=(), max_length=256)

    @field_validator("environments", "tools", mode="before")
    @classmethod
    def validate_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("environments", "tools")
    @classmethod
    def validate_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for item in value:
            _bounded_durable_text(item, info.field_name, max_chars=256, nonblank=True, clean=True)
        if value != tuple(sorted(set(value))):
            raise ValueError("Benchmark requirement names must be unique and sorted.")
        return value


class BenchmarkPackageV1(_PortableModel):
    """A versioned benchmark over the canonical authored-suite/case contracts."""

    schema_version: Literal[1] = 1
    revision: StrictStr
    id: StrictStr
    version: StrictStr
    scorer_id: StrictStr
    scorer_version: StrictStr
    suite: EvalSuiteDocumentV3
    scenarios: tuple[EvalScenarioDocumentV2, ...] = Field(default=(), max_length=1000)
    files: tuple[BenchmarkFileV1, ...] = Field(default=(), max_length=1000)
    requirements: BenchmarkRequirementsV1 = Field(default_factory=BenchmarkRequirementsV1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "scorer_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("version", "scorer_version")
    @classmethod
    def validate_versions(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value, info.field_name, max_chars=64, nonblank=True, clean=True
        )

    @field_validator("scenarios", "files", mode="before")
    @classmethod
    def validate_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> BenchmarkPackageV1:
        revisions = tuple(scenario.revision for scenario in self.scenarios)
        if revisions != tuple(sorted(set(revisions))):
            raise ValueError("Package scenarios must be unique and sorted by revision.")
        scenarios = {scenario.revision: scenario for scenario in self.scenarios}
        referenced = set()
        for case in self.suite.cases:
            if type(case.stimulus) is EvalScenarioStimulusV1:
                scenario = scenarios.get(case.stimulus.scenario_revision)
                if scenario is None or scenario.id != case.stimulus.scenario_id:
                    raise ValueError("Package is missing an exact case scenario.")
                referenced.add(scenario.revision)
        if referenced != set(scenarios):
            raise ValueError("Package contains unreferenced scenarios.")
        requirements = {}
        for scenario in self.scenarios:
            if scenario.target_key != self.suite.target_key:
                raise ValueError("Package scenarios and suite must use the same target key.")
            for requirement in scenario.artifact_requirements:
                if requirement.source != "fixture_digest":
                    raise ValueError("Package files must use portable fixture_digest requirements.")
                if requirement.size_bytes > BENCHMARK_FILE_MAX_BYTES:
                    raise ValueError("Benchmark file exceeds the per-file byte limit.")
                requirements[scenario.revision, requirement.id] = requirement
        keys = tuple((item.scenario_revision, item.requirement_id) for item in self.files)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Package file bindings must be unique and sorted.")
        if set(keys) != set(requirements):
            raise ValueError("Package files must cover exactly the scenario file requirements.")
        if sum(item.size_bytes for item in requirements.values()) > BENCHMARK_FILES_MAX_BYTES:
            raise ValueError("Benchmark files exceed the aggregate byte limit.")
        if self.revision != _model_content_revision(self, "benchmark package"):
            raise ValueError("Benchmark package revision does not match its contents.")
        if len(canonical_durable_json_bytes(self.model_dump(mode="json"), "benchmark package")) > (
            BENCHMARK_PACKAGE_MAX_BYTES
        ):
            raise ValueError("Benchmark package exceeds the manifest byte limit.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        version: str,
        scorer_id: str,
        scorer_version: str,
        suite: EvalSuiteDocumentV3,
        scenarios: Sequence[EvalScenarioDocumentV2] = (),
        files: Sequence[BenchmarkFileV1] = (),
        requirements: BenchmarkRequirementsV1 | None = None,
    ) -> BenchmarkPackageV1:
        material = {
            "schema_version": 1,
            "id": id,
            "version": version,
            "scorer_id": scorer_id,
            "scorer_version": scorer_version,
            "suite": suite.model_dump(mode="json"),
            "scenarios": [
                item.model_dump(mode="json") for item in sorted(scenarios, key=lambda s: s.revision)
            ],
            "files": [
                item.model_dump(mode="json")
                for item in sorted(files, key=lambda f: (f.scenario_revision, f.requirement_id))
            ],
            "requirements": (requirements or BenchmarkRequirementsV1()).model_dump(mode="json"),
        }
        return cls.model_validate(
            {"revision": _content_revision(material, "benchmark package"), **material}
        )


@dataclass(frozen=True)
class LoadedBenchmarkPackage:
    package: BenchmarkPackageV1
    root: Path
    # A bounded snapshot prevents source files changing between validation and publication.
    file_contents: tuple[tuple[BenchmarkFileV1, bytes], ...]
    manifest_path: Path | None = None


def benchmark_package_to_json(package: BenchmarkPackageV1) -> str:
    validated = BenchmarkPackageV1.model_validate(package.model_dump(mode="json"))
    return canonical_durable_json_bytes(
        validated.model_dump(mode="json"), "benchmark package"
    ).decode("utf-8")


def benchmark_package_from_json(source: str) -> BenchmarkPackageV1:
    if type(source) is not str or len(source) > BENCHMARK_PACKAGE_MAX_BYTES:
        raise ValueError("Benchmark manifest must be bounded UTF-8 JSON text.")
    if len(source.encode("utf-8")) > BENCHMARK_PACKAGE_MAX_BYTES:
        raise ValueError("Benchmark manifest exceeds the byte limit.")
    try:
        decoded = json.loads(
            source,
            parse_int=partial(parse_durable_json_integer_literal, field_name="benchmark package"),
            parse_constant=partial(
                reject_nonportable_json_constant, field_name="benchmark package"
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs, field_name="benchmark package"
            ),
        )
    except RecursionError as exc:
        raise ValueError("Benchmark manifest exceeds supported JSON nesting.") from exc
    if type(decoded) is not dict or type(decoded.get("schema_version")) is not int:
        raise ValueError("schema_version must be the integer 1.")
    return BenchmarkPackageV1.model_validate(copy_durable_json_object(decoded, "benchmark package"))


def load_benchmark_package(path: str | Path) -> LoadedBenchmarkPackage:
    """Read a manifest (or directory/benchmark.json) and verify every named input file."""

    source = Path(path)
    if source.is_dir():
        source = source / "benchmark.json"
    with source.open("rb") as stream:
        raw = stream.read(BENCHMARK_PACKAGE_MAX_BYTES + 1)
    if len(raw) > BENCHMARK_PACKAGE_MAX_BYTES:
        raise ValueError("Benchmark manifest exceeds the byte limit.")
    package = benchmark_package_from_json(raw.decode("utf-8"))
    root = source.resolve().parent
    scenarios = {scenario.revision: scenario for scenario in package.scenarios}
    contents = []
    for binding in package.files:
        file_path = root.joinpath(*PurePosixPath(binding.path).parts)
        current = file_path
        while current != root:
            if current.is_symlink():
                raise ValueError("Benchmark input paths must not contain symbolic links.")
            current = current.parent
        if not file_path.resolve().is_relative_to(root) or not file_path.is_file():
            raise ValueError("Benchmark input file is missing or outside its package.")
        requirement = next(
            item
            for item in scenarios[binding.scenario_revision].artifact_requirements
            if item.id == binding.requirement_id
        )
        with file_path.open("rb") as stream:
            content = stream.read(requirement.size_bytes + 1)
        if (
            len(content) != requirement.size_bytes
            or hashlib.sha256(content).hexdigest() != requirement.content_sha256
        ):
            raise ValueError("Benchmark input file does not match its immutable requirement.")
        contents.append((binding, content))
    return LoadedBenchmarkPackage(
        package=package, root=root, file_contents=tuple(contents), manifest_path=source.resolve()
    )


def _benchmark_source(package: BenchmarkPackageV1) -> EvaluationSourceIdentityV1:
    return EvaluationSourceIdentityV1(
        application_release_id=f"{package.id}@{package.version}",
        app_manifest_schema_version="benchmark-package-v1",
        app_manifest_fingerprint=package.revision[7:],
        evidence_revision=package.revision,
    )


def benchmark_package_scenarios(package: BenchmarkPackageV1) -> tuple[EvalScenarioDocumentV2, ...]:
    """Resolve scenarios with the same package provenance as their authored cases."""

    package = BenchmarkPackageV1.model_validate(package.model_dump(mode="json"))
    source = _benchmark_source(package)
    return tuple(
        EvalScenarioDocumentV2.create(
            id=scenario.id,
            target_key=scenario.target_key,
            name=scenario.name,
            description=scenario.description,
            source=source,
            events=scenario.events,
            artifact_requirements=scenario.artifact_requirements,
            secret_requirements=scenario.secret_requirements,
        )
        for scenario in package.scenarios
    )


def benchmark_suite_selection(
    package: BenchmarkPackageV1,
    case_ids: Sequence[str] | None = None,
) -> tuple[EvalSuiteDocumentV3, EvalSuiteSelectionV1]:
    """Bind package identity into the existing comparable case and cohort identities."""

    package = BenchmarkPackageV1.model_validate(package.model_dump(mode="json"))
    source = _benchmark_source(package)
    scenario_by_original_revision = dict(
        zip(
            (item.revision for item in package.scenarios),
            benchmark_package_scenarios(package),
            strict=True,
        )
    )
    draft = EvalSuiteDraftV3.from_document(package.suite)
    cases = []
    for case in draft.cases:
        material = {**case.model_dump(mode="json"), "source": source.model_dump(mode="json")}
        if type(case.stimulus) is EvalScenarioStimulusV1:
            scenario = scenario_by_original_revision[case.stimulus.scenario_revision]
            material["stimulus"] = EvalScenarioStimulusV1(
                scenario_id=scenario.id, scenario_revision=scenario.revision
            ).model_dump(mode="json")
        cases.append(EvalCaseDraftV2.model_validate(material))
    document = compile_eval_suite_draft_v3(draft.model_copy(update={"cases": tuple(cases)}))
    return document, eval_suite_selection(document, case_ids)
