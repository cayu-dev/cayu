from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation

from cayu import (
    AgentSpec,
    Environment,
    EnvironmentSpec,
    LocalArtifactStore,
    LocalWorkspace,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    WorkspaceReadResult,
)
from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.artifacts import ArtifactListResult, ArtifactMetadata, ArtifactReadResult, ArtifactScope
from cayu.core.events import Event, EventType
from cayu.evals.corpus import (
    ArtifactAssertionSpec,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    RunInputSpec,
    TrialRequestSpec,
    WorkspaceFileAssertionSpec,
)
from cayu.evals.evidence import ArtifactStructuralEvidenceV1, project_assertion_evidence_view
from cayu.evals.execution import CorpusTarget, run_corpus_suite
from cayu.evals.execution_comparison import (
    compare_corpus_execution_results,
    corpus_execution_compatibility,
)
from cayu.evals.execution_reporting import (
    corpus_execution_result_to_json,
    render_corpus_execution_comparison_html,
    render_corpus_execution_html,
)
from cayu.evals.models import (
    ARTIFACT_PROBE_MAX_BYTES,
    ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
    WORKSPACE_PROBE_MAX_BYTES,
    ArtifactContentProbe,
    ArtifactProbeRequirement,
    EvalAssertionResult,
    EvalOutcome,
    ProbeRequirements,
    Trajectory,
    TrajectoryProbes,
    WorkspaceStructuralProbe,
)
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.evals.published import (
    PublishedArtifactDetail,
    PublishedWorkspaceFileDetail,
    _published_detail,
)
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.runner import _capture_probes
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.usage import session_usage_summary

_STRUCTURAL_CONTENT = b'{"source":"structural-eval","status":"ready"}\n'


class _StructuralOutputTool(Tool):
    spec = ToolSpec(
        name="write_structural_output",
        description="Write deterministic workspace and artifact output.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
        workspace_mutation=True,
    )

    def __init__(self, content: bytes = _STRUCTURAL_CONTENT) -> None:
        super().__init__(self.spec)
        self.content = content

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        if ctx.workspace is None or ctx.artifact_store is None:
            return ToolResult(content="Structural storage is unavailable.", is_error=True)
        await ctx.workspace.write_bytes("outputs/result.json", self.content)
        await ctx.artifact_store.put_bytes(
            self.content,
            artifact_id=("art_" + hashlib.sha256(ctx.session_id.encode("utf-8")).hexdigest()[:32]),
            filename="result.json",
            content_type="application/json",
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
        )
        return ToolResult(content="Structural output written.")


def structural_corpus(*, target_key: str = "structural-agent") -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id="structural-contract",
        name="Structural contract",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="structural-output",
        suite_id=suite.id,
        name="Structural output",
        source=EvaluationSourceIdentityV1(
            application_release_id="structural-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "b" * 64,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Write the structural output."),)),
        assertions=(
            WorkspaceFileAssertionSpec(
                id="workspace-output",
                path="outputs/result.json",
                minimum_bytes=1,
                sha256=hashlib.sha256(_STRUCTURAL_CONTENT).hexdigest(),
            ),
            ArtifactAssertionSpec(
                id="artifact-output",
                scope="session",
                filename="result.json",
                content_type="application/json",
                minimum_bytes=1,
                sha256=hashlib.sha256(_STRUCTURAL_CONTENT).hexdigest(),
                text_contains='"status":"ready"',
                min_count=1,
                max_count=1,
            ),
        ),
    )
    policy = EvaluationEvidencePolicySpec.create(include_artifact_text=True)
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=policy,
        suites=(suite,),
        cases=(case,),
    )


def structural_target(
    root: Path,
    *,
    target_key: str = "structural-agent",
    content: bytes = _STRUCTURAL_CONTENT,
    application_release_id: str = "structural-release",
) -> CorpusTarget:
    app = CayuApp(enable_logging=False)
    workspace_root = root / "workspace"
    workspace_root.mkdir(parents=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(workspace_root, workspace_id="structural-workspace"),
            artifact_store=LocalArtifactStore(
                root / "artifacts",
                store_id="structural-artifacts",
            ),
        ),
        default=True,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="structural-call",
                        name="write_structural_output",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Structural output complete."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="agent", model="fixture-model"),
        tools=[_StructuralOutputTool(content)],
    )
    return CorpusTarget(
        key=target_key,
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[]),
        application_release_id=application_release_id,
        evidence_policy=EvaluationEvidencePolicySpec.create(include_artifact_text=True),
    )


def _session() -> Session:
    return Session(
        id="structural-session",
        agent_name="agent",
        provider_name="fixture",
        model="fixture-model",
        environment_name="local",
        invocation=fixture_session_invocation("structural-session"),
        status=SessionStatus.COMPLETED,
    )


def _evidence(probes: TrajectoryProbes, *, include_artifact_text: bool = False):
    events = (Event(type=EventType.SESSION_COMPLETED, session_id="structural-session"),)
    return project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        Trajectory(
            session=_session(),
            events=events,
            usage_summary=session_usage_summary("structural-session", events),
            probes=probes,
        ),
        evidence_policy=EvaluationEvidencePolicySpec.create(
            include_artifact_text=include_artifact_text
        ),
    )


def test_structural_specs_reject_unsafe_and_contradictory_contracts() -> None:
    for path in (
        "../secret",
        "/absolute",
        "a/../b",
        "a\\b",
        "C:/secret",
        "report:stream",
        "CON.txt",
        "output./report.txt",
        "cafe\N{COMBINING ACUTE ACCENT}.txt",
    ):
        with pytest.raises(ValidationError):
            WorkspaceFileAssertionSpec(id="workspace", path=path)
    with pytest.raises(ValidationError, match="Absent workspace"):
        WorkspaceFileAssertionSpec(
            id="workspace",
            path="out.txt",
            present=False,
            minimum_bytes=1,
        )
    with pytest.raises(ValidationError, match="maximum_bytes"):
        ArtifactAssertionSpec(
            id="artifact",
            minimum_bytes=2,
            maximum_bytes=1,
        )
    with pytest.raises(ValidationError, match="max_count"):
        ArtifactAssertionSpec(id="artifact", min_count=2, max_count=1)
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        ArtifactAssertionSpec(
            id="artifact",
            text_contains="😀" * 17_000,
        )


def test_published_structural_details_reject_contradictory_observations() -> None:
    with pytest.raises(ValidationError, match="present workspace observations"):
        PublishedWorkspaceFileDetail(
            path="out.txt",
            expected_present=True,
            digest_required=False,
            observation_state="available",
            actual_present=False,
            actual_size_bytes=1,
        )
    with pytest.raises(ValidationError, match="digest expectation"):
        PublishedWorkspaceFileDetail(
            path="out.txt",
            expected_present=True,
            digest_required=False,
            observation_state="available",
            actual_present=True,
            actual_size_bytes=1,
            digest_matched=True,
        )
    with pytest.raises(ValidationError, match="max_count"):
        PublishedArtifactDetail(
            scope="session",
            digest_required=False,
            text_required=False,
            observation_state="unavailable",
            min_count=2,
            max_count=1,
        )

    with pytest.raises(ValidationError, match="matching count"):
        PublishedArtifactDetail(
            scope="session",
            digest_required=False,
            text_required=True,
            observation_state="truncated",
            min_count=1,
            matching_count=1,
        )


def test_published_structural_details_preserve_typed_unavailable_reason() -> None:
    detail = _published_detail(
        ArtifactAssertionSpec(
            id="artifact",
            filename="report.txt",
            text_contains="ready",
        ),
        EvalAssertionResult(
            name="artifact",
            outcome=EvalOutcome.UNAVAILABLE,
            message="artifact text evidence is truncated.",
            metadata={"evidence_area": "artifact text", "evidence_state": "truncated"},
        ),
    )

    assert type(detail) is PublishedArtifactDetail
    assert detail.observation_state == "truncated"
    assert detail.matching_count is None


def test_structural_probe_models_enforce_fixed_read_and_text_boundaries() -> None:
    with pytest.raises(ValidationError, match="fixed read limit"):
        WorkspaceStructuralProbe(
            state="present",
            total_bytes=WORKSPACE_PROBE_MAX_BYTES + 1,
            digest_state="complete",
            sha256="0" * 64,
        )

    oversized = ArtifactMetadata(
        id="oversized-artifact",
        filename="large.json",
        content_type="application/json",
        size_bytes=ARTIFACT_PROBE_MAX_BYTES + 1,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    with pytest.raises(ValidationError, match="fixed read limit"):
        TrajectoryProbes(
            artifacts_available=True,
            artifact_scopes_captured=(ArtifactScope.SESSION,),
            artifacts=(oversized,),
            artifact_content_probes=(
                ArtifactContentProbe(
                    artifact_id=oversized.id,
                    digest_state="complete",
                    sha256="0" * 64,
                ),
            ),
        )

    binary = oversized.model_copy(
        update={
            "id": "binary-artifact",
            "content_type": "application/octet-stream",
            "size_bytes": ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
        }
    )
    with pytest.raises(ValidationError, match="supported textual content"):
        TrajectoryProbes(
            artifacts_available=True,
            artifact_scopes_captured=(ArtifactScope.SESSION,),
            artifacts=(binary,),
            artifact_content_probes=(
                ArtifactContentProbe(
                    artifact_id=binary.id,
                    digest_state="complete",
                    sha256="0" * 64,
                    text_state="available",
                    text="binary",
                ),
            ),
        )

    textual = binary.model_copy(
        update={
            "id": "text-artifact",
            "content_type": "text/plain",
            "size_bytes": 10,
        }
    )
    with pytest.raises(ValidationError, match="complete bytes"):
        TrajectoryProbes(
            artifacts_available=True,
            artifact_scopes_captured=(ArtifactScope.SESSION,),
            artifacts=(textual,),
            artifact_content_probes=(
                ArtifactContentProbe(
                    artifact_id=textual.id,
                    digest_state="complete",
                    sha256="0" * 64,
                    text_state="available",
                    text="short",
                ),
            ),
        )

    with pytest.raises(ValidationError, match="canonical POSIX"):
        TrajectoryProbes(
            workspace_available=True,
            workspace_structures={
                "output//report.txt": WorkspaceStructuralProbe(
                    state="missing",
                    digest_state="unavailable",
                )
            },
        )

    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        ArtifactStructuralEvidenceV1(
            observation_index=1,
            scope="session",
            filename="large.txt",
            content_type="text/plain",
            size_bytes=ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
            digest_state="complete",
            sha256="0" * 64,
            text_state="available",
            text="é" * (ARTIFACT_PUBLIC_TEXT_MAX_BYTES // 2 + 1),
        )

    with pytest.raises(ValidationError, match="less than or equal"):
        WorkspaceStructuralProbe(
            state="present",
            total_bytes=MAX_PORTABLE_JSON_INTEGER + 1,
            digest_state="limit_exceeded",
        )

    with pytest.raises(ValidationError, match="less than or equal"):
        ArtifactStructuralEvidenceV1(
            observation_index=1,
            scope="session",
            filename="nonportable-size.bin",
            content_type="application/octet-stream",
            size_bytes=MAX_PORTABLE_JSON_INTEGER + 1,
            digest_state="limit_exceeded",
            text_state="unsupported",
        )


def test_workspace_structure_evaluates_presence_size_digest_and_absence() -> None:
    content = b"complete output"
    evidence = _evidence(
        TrajectoryProbes(
            workspace_available=True,
            workspace_structures={
                "out.txt": WorkspaceStructuralProbe(
                    state="present",
                    total_bytes=len(content),
                    digest_state="complete",
                    sha256=hashlib.sha256(content).hexdigest(),
                ),
                "missing.txt": WorkspaceStructuralProbe(
                    state="missing",
                    digest_state="unavailable",
                ),
            },
        )
    )
    matched = evaluate_assertion_spec(
        WorkspaceFileAssertionSpec(
            id="workspace",
            path="out.txt",
            minimum_bytes=1,
            maximum_bytes=100,
            sha256=hashlib.sha256(content).hexdigest(),
        ),
        evidence,
    )
    absent = evaluate_assertion_spec(
        WorkspaceFileAssertionSpec(id="missing", path="missing.txt", present=False),
        evidence,
    )
    assert matched.outcome is EvalOutcome.PASSED
    assert absent.outcome is EvalOutcome.PASSED
    assert "complete output" not in evidence.model_dump_json()


def test_workspace_partial_digest_is_unavailable_without_prefix_identity() -> None:
    evidence = _evidence(
        TrajectoryProbes(
            workspace_available=True,
            workspace_structures={
                "large.bin": WorkspaceStructuralProbe(
                    state="present",
                    total_bytes=2_000_000,
                    digest_state="limit_exceeded",
                )
            },
        )
    )
    result = evaluate_assertion_spec(
        WorkspaceFileAssertionSpec(id="workspace", path="large.bin", sha256="0" * 64),
        evidence,
    )
    assert result.outcome is EvalOutcome.UNAVAILABLE
    assert result.score is None


def test_artifact_structure_and_opt_in_text_evaluate_without_private_identity() -> None:
    content = b"public report: ready"
    metadata = ArtifactMetadata(
        id="private-artifact-id",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=len(content),
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    probes = TrajectoryProbes(
        artifacts_available=True,
        artifact_scopes_captured=(ArtifactScope.SESSION,),
        artifacts=(metadata,),
        artifact_content_probes=(
            ArtifactContentProbe(
                artifact_id=metadata.id,
                digest_state="complete",
                sha256=hashlib.sha256(content).hexdigest(),
                text_state="available",
                text=content.decode(),
            ),
        ),
    )
    evidence = _evidence(probes, include_artifact_text=True)
    result = evaluate_assertion_spec(
        ArtifactAssertionSpec(
            id="artifact",
            filename="report.txt",
            content_type="text/plain",
            minimum_bytes=1,
            sha256=hashlib.sha256(content).hexdigest(),
            text_contains="ready",
        ),
        evidence,
    )
    serialized = evidence.model_dump_json()
    assert result.outcome is EvalOutcome.PASSED
    assert "private-artifact-id" not in serialized
    assert "public report: ready" in serialized


def test_artifact_text_requires_server_published_policy() -> None:
    spec = ArtifactAssertionSpec(id="artifact", filename="report.txt", text_contains="ready")
    with pytest.raises(ValueError, match="explicitly retained artifact text"):
        compile_assertion_spec(
            spec,
            app=CayuApp(enable_logging=False),
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            trusted_pricing=None,
        )


def test_artifact_missing_and_incomplete_scope_evidence_are_distinct() -> None:
    spec = ArtifactAssertionSpec(id="artifact", filename="report.txt")
    missing = evaluate_assertion_spec(
        spec,
        _evidence(
            TrajectoryProbes(
                artifacts_available=True,
                artifact_scopes_captured=(ArtifactScope.SESSION,),
            )
        ),
    )
    unavailable = evaluate_assertion_spec(
        spec,
        _evidence(
            TrajectoryProbes(
                artifacts_available=True,
                artifact_scopes_unavailable=(ArtifactScope.SESSION,),
            )
        ),
    )
    limited = evaluate_assertion_spec(
        spec,
        _evidence(
            TrajectoryProbes(
                artifacts_available=True,
                artifact_scopes_truncated=(ArtifactScope.SESSION,),
            )
        ),
    )

    assert missing.outcome is EvalOutcome.FAILED
    assert missing.metadata["matching_count"] == 0
    assert unavailable.outcome is EvalOutcome.UNAVAILABLE
    assert limited.outcome is EvalOutcome.UNAVAILABLE


@pytest.mark.parametrize(
    ("text_state", "content_type"),
    [
        ("unavailable", "application/json"),
        ("unsupported", "application/octet-stream"),
        ("truncated", "application/json"),
        ("redacted", "application/json"),
        ("malformed", "application/json"),
    ],
)
def test_incomplete_artifact_text_never_becomes_a_candidate_mismatch(
    text_state: str,
    content_type: str,
) -> None:
    metadata = ArtifactMetadata(
        id="private-artifact-id",
        filename="report.txt",
        content_type=content_type,
        size_bytes=10,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    evidence = _evidence(
        TrajectoryProbes(
            artifacts_available=True,
            artifact_scopes_captured=(ArtifactScope.SESSION,),
            artifacts=(metadata,),
            artifact_content_probes=(
                ArtifactContentProbe(
                    artifact_id=metadata.id,
                    digest_state="complete",
                    sha256="0" * 64,
                    text_state=text_state,
                ),
            ),
        ),
        include_artifact_text=True,
    )

    result = evaluate_assertion_spec(
        ArtifactAssertionSpec(
            id="artifact",
            filename="report.txt",
            text_contains="ready",
        ),
        evidence,
    )

    assert result.outcome is EvalOutcome.UNAVAILABLE
    assert "matching_count" not in result.metadata


def test_partial_artifact_digest_is_not_compared_as_whole_object_identity() -> None:
    metadata = ArtifactMetadata(
        id="private-artifact-id",
        filename="report.bin",
        content_type="application/octet-stream",
        size_bytes=2_000_000,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    result = evaluate_assertion_spec(
        ArtifactAssertionSpec(
            id="artifact",
            filename="report.bin",
            sha256="0" * 64,
        ),
        _evidence(
            TrajectoryProbes(
                artifacts_available=True,
                artifact_scopes_captured=(ArtifactScope.SESSION,),
                artifacts=(metadata,),
                artifact_content_probes=(
                    ArtifactContentProbe(
                        artifact_id=metadata.id,
                        digest_state="limit_exceeded",
                        text_state="unavailable",
                    ),
                ),
            )
        ),
    )

    assert result.outcome is EvalOutcome.UNAVAILABLE


class _ProbeWorkspace:
    async def read_bytes(self, path: str, *, max_bytes: int | None = None):
        content = b"workspace output"
        selected = content if max_bytes is None else content[:max_bytes]
        return WorkspaceReadResult(
            content=selected,
            total_bytes=len(content),
            truncated=len(selected) < len(content),
        )


class _ProbeArtifactStore:
    def __init__(self, metadata: ArtifactMetadata, content: bytes) -> None:
        self.metadata = metadata
        self.content = content
        self.read_ids: list[str] = []
        self.list_filters: list[tuple[ArtifactScope, str | None, str | None, int | None]] = []

    async def list(self, *, scope, session_id=None, environment_name=None, limit=None):
        self.list_filters.append((scope, session_id, environment_name, limit))
        return ArtifactListResult(artifacts=(self.metadata,), total_count=1)

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        self.read_ids.append(artifact_id)
        selected = self.content if max_bytes is None else self.content[:max_bytes]
        return ArtifactReadResult(
            metadata=self.metadata,
            content=selected,
            total_bytes=len(self.content),
            truncated=len(selected) < len(self.content),
        )


def test_capture_fails_closed_for_nonportable_structural_byte_counts() -> None:
    class _NonportableSizeWorkspace:
        async def read_bytes(self, path: str, *, max_bytes: int | None = None):
            del path, max_bytes
            return WorkspaceReadResult(
                content=b"x",
                total_bytes=MAX_PORTABLE_JSON_INTEGER + 1,
                truncated=True,
            )

    metadata = ArtifactMetadata(
        id="nonportable-size-artifact",
        filename="large.bin",
        content_type="application/octet-stream",
        size_bytes=MAX_PORTABLE_JSON_INTEGER + 1,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    store = _ProbeArtifactStore(metadata, b"")

    class _ProbeApp:
        def get_environment(self, name):
            del name
            return SimpleNamespace(
                environment=SimpleNamespace(
                    workspace=_NonportableSizeWorkspace(),
                    artifact_store=store,
                )
            )

    probes = asyncio.run(
        _capture_probes(
            _ProbeApp(),
            _session(),
            ProbeRequirements(
                workspace_structure_paths=frozenset({"out.txt"}),
                artifact_requirements=(
                    ArtifactProbeRequirement(
                        scope=ArtifactScope.SESSION,
                        filename="large.bin",
                    ),
                ),
            ),
        )
    )

    workspace = probes.workspace_structures["out.txt"]
    assert workspace.state == "unavailable"
    assert workspace.total_bytes is None

    evidence = _evidence(probes)
    assert evidence.workspace_evidence_state == "complete"
    assert evidence.workspace_files[0].state == "unavailable"
    assert evidence.artifacts == ()
    assert tuple((item.scope, item.state) for item in evidence.artifact_scopes) == (
        ("session", "unavailable"),
    )
    assert str(MAX_PORTABLE_JSON_INTEGER + 1) not in evidence.model_dump_json()


def test_capture_reads_only_declared_structural_paths_and_prefiltered_artifacts() -> None:
    content = b"public report"
    metadata = ArtifactMetadata(
        id="report-artifact",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=len(content),
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    store = _ProbeArtifactStore(metadata, content)

    class _ProbeApp:
        def get_environment(self, name):
            return SimpleNamespace(
                environment=SimpleNamespace(
                    workspace=_ProbeWorkspace(),
                    artifact_store=store,
                )
            )

        def redact_utf8_head(self, value, *, max_bytes, source_complete):
            text = value.decode()
            return text, len(value) > max_bytes or not source_complete

    probes = asyncio.run(
        _capture_probes(
            _ProbeApp(),
            _session(),
            ProbeRequirements(
                workspace_structure_paths=frozenset({"out.txt"}),
                artifact_requirements=(
                    ArtifactProbeRequirement(
                        scope=ArtifactScope.SESSION,
                        filename="report.txt",
                        capture_digest=True,
                        capture_text=True,
                    ),
                ),
            ),
        )
    )
    assert probes.workspace_files == {}
    assert (
        probes.workspace_structures["out.txt"].sha256
        == hashlib.sha256(b"workspace output").hexdigest()
    )
    assert store.read_ids == ["report-artifact"]
    assert store.list_filters == [(ArtifactScope.SESSION, "structural-session", None, 256)]
    assert probes.artifact_content_probes[0].text == "public report"


def test_capture_rejects_incomplete_backend_progress_and_oversized_listings() -> None:
    class _IncompleteWorkspace:
        async def read_bytes(self, path: str, *, max_bytes: int | None = None):
            del path, max_bytes
            return WorkspaceReadResult(
                content=b"x",
                total_bytes=2,
                truncated=False,
                source_bytes_read=2,
            )

    class _WorkspaceApp:
        def get_environment(self, name):
            del name
            return SimpleNamespace(
                environment=SimpleNamespace(workspace=_IncompleteWorkspace(), artifact_store=None)
            )

    workspace_probes = asyncio.run(
        _capture_probes(
            _WorkspaceApp(),
            _session(),
            ProbeRequirements(workspace_structure_paths=frozenset({"out.txt"})),
        )
    )
    assert workspace_probes.workspace_structures["out.txt"].state == "unavailable"

    metadata = ArtifactMetadata(
        id="report-artifact",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=6,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )

    class _IncompleteArtifactStore(_ProbeArtifactStore):
        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            del artifact_id, max_bytes
            return ArtifactReadResult(
                metadata=self.metadata,
                content=b"r",
                total_bytes=6,
                truncated=False,
                source_bytes_read=6,
            )

    incomplete_store = _IncompleteArtifactStore(metadata, b"report")

    class _ArtifactApp:
        def __init__(self, store) -> None:
            self.store = store

        def get_environment(self, name):
            del name
            return SimpleNamespace(
                environment=SimpleNamespace(workspace=None, artifact_store=self.store)
            )

    artifact_requirement = ProbeRequirements(
        artifact_requirements=(
            ArtifactProbeRequirement(
                scope=ArtifactScope.SESSION,
                filename="report.txt",
                capture_digest=True,
            ),
        )
    )
    incomplete_probes = asyncio.run(
        _capture_probes(_ArtifactApp(incomplete_store), _session(), artifact_requirement)
    )
    assert incomplete_probes.artifact_content_probes[0].digest_state == "unavailable"

    class _OversizedListStore(_ProbeArtifactStore):
        async def list(self, *, scope, session_id=None, environment_name=None, limit=None):
            del environment_name, limit
            artifacts = tuple(
                self.metadata.model_copy(update={"id": f"report-{index}"}) for index in range(257)
            )
            return ArtifactListResult(artifacts=artifacts, total_count=len(artifacts))

    oversized_store = _OversizedListStore(metadata, b"report")
    oversized_probes = asyncio.run(
        _capture_probes(_ArtifactApp(oversized_store), _session(), artifact_requirement)
    )
    assert oversized_probes.artifact_scopes_unavailable == (ArtifactScope.SESSION,)
    assert oversized_probes.artifacts == ()


def test_capture_turns_artifact_redaction_failure_into_unavailable_evidence() -> None:
    content = b"public report"
    metadata = ArtifactMetadata(
        id="report-artifact",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=len(content),
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    store = _ProbeArtifactStore(metadata, content)

    class _FailingRedactorApp:
        def get_environment(self, name):
            del name
            return SimpleNamespace(
                environment=SimpleNamespace(workspace=None, artifact_store=store)
            )

        def redact_utf8_head(self, value, *, max_bytes, source_complete):
            del value, max_bytes, source_complete
            raise RuntimeError("redactor unavailable")

    probes = asyncio.run(
        _capture_probes(
            _FailingRedactorApp(),
            _session(),
            ProbeRequirements(
                artifact_requirements=(
                    ArtifactProbeRequirement(
                        scope=ArtifactScope.SESSION,
                        filename="report.txt",
                        capture_text=True,
                    ),
                )
            ),
        )
    )

    assert probes.artifact_content_probes[0].text_state == "unavailable"


def test_capture_rejects_artifact_read_metadata_that_changed_after_listing() -> None:
    listed = ArtifactMetadata(
        id="report-artifact",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=6,
        scope=ArtifactScope.SESSION,
        session_id="structural-session",
    )
    changed = listed.model_copy(update={"filename": "different.txt"})

    class _ChangedReadStore(_ProbeArtifactStore):
        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            result = await super().read_bytes(artifact_id, max_bytes=max_bytes)
            return ArtifactReadResult(
                metadata=changed,
                content=result.content,
                total_bytes=result.total_bytes,
                truncated=result.truncated,
            )

    store = _ChangedReadStore(listed, b"report")

    class _ProbeApp:
        def get_environment(self, name):
            return SimpleNamespace(
                environment=SimpleNamespace(workspace=None, artifact_store=store)
            )

    probes = asyncio.run(
        _capture_probes(
            _ProbeApp(),
            _session(),
            ProbeRequirements(
                artifact_requirements=(
                    ArtifactProbeRequirement(
                        scope=ArtifactScope.SESSION,
                        filename="report.txt",
                        capture_digest=True,
                    ),
                )
            ),
        )
    )

    assert probes.artifacts == (listed,)
    assert probes.artifact_content_probes == (
        ArtifactContentProbe(
            artifact_id=listed.id,
            digest_state="unavailable",
            text_state="unavailable",
        ),
    )


def test_capture_does_not_turn_an_invalid_scoped_listing_into_observed_absence() -> None:
    unrelated = ArtifactMetadata(
        id="unrelated-artifact",
        filename="report.txt",
        content_type="text/plain",
        size_bytes=6,
        scope=ArtifactScope.SESSION,
        session_id="another-session",
    )
    store = _ProbeArtifactStore(unrelated, b"report")

    class _ProbeApp:
        def get_environment(self, name):
            return SimpleNamespace(
                environment=SimpleNamespace(workspace=None, artifact_store=store)
            )

    probes = asyncio.run(
        _capture_probes(
            _ProbeApp(),
            _session(),
            ProbeRequirements(
                artifact_requirements=(
                    ArtifactProbeRequirement(
                        scope=ArtifactScope.SESSION,
                        filename="report.txt",
                        capture_digest=True,
                    ),
                )
            ),
        )
    )

    assert probes.artifact_scopes_captured == ()
    assert probes.artifact_scopes_unavailable == (ArtifactScope.SESSION,)
    assert probes.artifacts == ()
    assert store.read_ids == []


def test_structural_assertions_run_publish_present_report_and_compare_safely(tmp_path) -> None:
    corpus = structural_corpus()
    baseline = asyncio.run(
        run_corpus_suite(
            structural_target(tmp_path / "baseline"),
            corpus,
            corpus.suites[0].id,
        )
    )

    assert baseline.run.status == "passed"
    trial = baseline.run.cases[0].trials[0]
    assert [assertion.detail.kind for assertion in trial.assertions] == [
        "workspace_file",
        "artifact",
    ]
    workspace_detail = trial.assertions[0].detail
    artifact_detail = trial.assertions[1].detail
    assert workspace_detail.actual_present is True
    assert workspace_detail.observation_state == "available"
    assert workspace_detail.actual_size_bytes == len(_STRUCTURAL_CONTENT)
    assert workspace_detail.digest_matched is True
    assert artifact_detail.matching_count == 1
    assert artifact_detail.observation_state == "available"
    assert artifact_detail.text_required is True

    presentation = present_eval_result(baseline)
    presented_assertions = presentation.cases[0].trials[0].assertions
    assert [item.structure.kind for item in presented_assertions] == [
        "workspace_file",
        "artifact",
    ]

    serialized = corpus_execution_result_to_json(baseline)
    rendered = render_corpus_execution_html(baseline)
    for forbidden in (
        '"artifact_id"',
        '"store_id"',
        "structural-artifacts",
        '"source":"structural-eval"',
    ):
        assert forbidden not in serialized
        assert forbidden not in rendered
    assert '"kind": "workspace_file"' in serialized
    assert '"kind": "artifact"' in serialized
    assert "outputs/result.json" in rendered
    assert "result.json" in rendered

    current = asyncio.run(
        run_corpus_suite(
            structural_target(
                tmp_path / "current",
                content=b'{"source":"structural-eval","status":"not-ready"}\n',
                application_release_id="structural-release-2",
            ),
            corpus,
            corpus.suites[0].id,
        )
    )
    comparison = compare_corpus_execution_results(baseline, current)
    assert comparison.compatibility.comparable is True
    assert comparison.regressions
    comparison_html = render_corpus_execution_comparison_html(comparison)
    assert "passed → failed" in comparison_html
    assert '"artifact_id"' not in comparison_html

    original_case = corpus.cases[0]
    changed_case = EvalCaseSpec.create(
        id=original_case.id,
        suite_id=original_case.suite_id,
        name=original_case.name,
        source=original_case.source,
        input=original_case.input,
        assertions=(
            WorkspaceFileAssertionSpec(
                id="workspace-output",
                path="outputs/other.json",
            ),
            original_case.assertions[1],
        ),
    )
    changed_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        suites=corpus.suites,
        cases=(changed_case,),
    )
    changed_contract = asyncio.run(
        run_corpus_suite(
            structural_target(tmp_path / "changed-contract"),
            changed_corpus,
            changed_corpus.suites[0].id,
        )
    )
    compatibility = corpus_execution_compatibility(baseline, changed_contract)
    assert compatibility.comparable is False
    assert {reason.value for reason in compatibility.reasons} == {
        "corpus_revision_mismatch",
        "case_contract_mismatch",
        "assertion_contract_mismatch",
    }


def test_captured_session_scores_reviewed_structural_assertions_without_private_identity() -> None:
    from tests.evals.test_session_promotion import _run_trajectory

    from cayu import (
        InMemorySessionStore,
        build_captured_evaluation_candidate,
        score_captured_evaluation_candidate,
    )

    app, base_trajectory = asyncio.run(_run_trajectory(InMemorySessionStore()))
    assert base_trajectory.session is not None
    workspace_content = b"captured workspace"
    artifact_content = b'{"status":"ready"}'
    artifact = ArtifactMetadata(
        id="private-captured-artifact",
        filename="captured.json",
        content_type="application/json",
        size_bytes=len(artifact_content),
        scope=ArtifactScope.SESSION,
        session_id=base_trajectory.session.id,
    )
    trajectory_document = base_trajectory.model_dump(
        mode="python", round_trip=True, warnings="none"
    )
    trajectory_document["probes"] = TrajectoryProbes(
        workspace_available=True,
        workspace_structures={
            "outputs/captured.txt": WorkspaceStructuralProbe(
                state="present",
                total_bytes=len(workspace_content),
                digest_state="complete",
                sha256=hashlib.sha256(workspace_content).hexdigest(),
            )
        },
        artifacts_available=True,
        artifact_scopes_captured=(ArtifactScope.SESSION,),
        artifacts=(artifact,),
        artifact_content_probes=(
            ArtifactContentProbe(
                artifact_id=artifact.id,
                digest_state="complete",
                sha256=hashlib.sha256(artifact_content).hexdigest(),
                text_state="available",
                text=artifact_content.decode(),
            ),
        ),
    )
    trajectory = Trajectory.model_validate(trajectory_document)
    policy = EvaluationEvidencePolicySpec.create(include_artifact_text=True)
    candidate = build_captured_evaluation_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="captured-structural-release",
        evidence_policy=policy,
    )
    edited_case = type(candidate.case).create(
        id=candidate.case.id,
        suite_id=candidate.case.suite_id,
        name=candidate.case.name,
        description=candidate.case.description,
        source=candidate.case.source,
        input=None,
        assertions=(
            WorkspaceFileAssertionSpec(
                id="workspace",
                path="outputs/captured.txt",
                sha256=hashlib.sha256(workspace_content).hexdigest(),
            ),
            ArtifactAssertionSpec(
                id="artifact",
                filename="captured.json",
                content_type="application/json",
                sha256=hashlib.sha256(artifact_content).hexdigest(),
                text_contains="ready",
                min_count=1,
                max_count=1,
            ),
        ),
    )
    reviewed = type(candidate).create(
        target_key=candidate.target_key,
        source=candidate.source,
        evidence_policy=candidate.evidence_policy,
        pricing_profile=candidate.pricing_profile,
        evidence=candidate.evidence,
        suite=candidate.suite,
        case=edited_case,
    )

    score = score_captured_evaluation_candidate(
        app,
        trajectory,
        reviewed,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="captured-structural-release",
    )

    assert score.status == "passed"
    assert [item.outcome for item in score.assertions] == ["passed", "passed"]
    assert "private-captured-artifact" not in reviewed.model_dump_json()
