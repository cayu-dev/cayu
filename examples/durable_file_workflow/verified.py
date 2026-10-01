"""Contract-verified file workflow: rejection, bounded continuation, acceptance, recovery.

Run with:
    uv run python -m examples.durable_file_workflow.verified
    uv run python -m examples.durable_file_workflow.verified --store sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import secrets
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from examples.durable_file_workflow.file_worker import (
    AGENT_NAME,
    ARTIFACT,
    PROGRAM,
    register_file_worker,
)
from pydantic import SecretStr

from cayu import (
    CayuApp,
    CompletionConstraintOutcome,
    CompletionContinuationPolicy,
    CompletionCriterionOutcome,
    CompletionDecision,
    CompletionDecisionApplicationReceipt,
    CompletionGap,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionResultReference,
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverRef,
    CompletionResultResolverRequest,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerifierDecision,
    CompletionVerifierProfileComponentDeclaration,
    CompletionVerifierRef,
    CompletionVerifierRequest,
    CriterionOutcomeStatus,
    DeterministicCompletionVerifier,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventQuery,
    EventType,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    InMemoryTaskStore,
    LocalRunner,
    LocalWorkspace,
    Message,
    NativeBinding,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RunLimits,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TextPart,
    VerifiedTaskHandler,
    VerifiedTaskHandlerReport,
    VerifiedTaskPreparationContext,
    VerifiedTaskProposalContext,
    VerifiedTaskWorker,
    VerifiedTaskWorkerDraining,
    WorkAttemptProposalRequest,
    WorkConstraint,
    WorkContractDraft,
    WorkCriterion,
    WorkEvidenceReference,
    WorkEvidenceRequirement,
    completion_result_sha256,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

TASK_ID = "transform-source"
SOURCE_TEXT = "cayu"
SOURCE_FILE = "source.txt"
# Workspace files are worker-written, so every read is bounded.
MAX_FILE_BYTES = 64 * 1024
CONTINUATION_TYPE = "cayu.verified-task-continuation.v1"
MISSING_NEWLINE = "artifact.missing_trailing_newline"
EXTRA_NEWLINE = "artifact.extra_trailing_newline"
CONTENT_MISMATCH = "artifact.content_mismatch"
ARTIFACT_UNAVAILABLE = "artifact.unavailable"
INPUT_MODIFIED = "input.modified"
INPUT_UNAVAILABLE = "input.unavailable"
SOURCE_UNAVAILABLE = "source.unavailable"
RESULT_MISMATCH = "result.mismatch"

GOAL_PROMPT = """\
Goal: Transform the supplied source text into the required artifact.
Inputs: source.txt in the session workspace. Do not modify it.
Output contract: result.txt contains the upper-cased source followed by exactly one newline.
Constraints: Work only in the session workspace. Use process commands, never shell commands.
An independent verifier decides completion. When you receive its gaps, repair them.
"""

# Attempt 1 omits the trailing newline; the repair restores it. Each run appends
# to runs.log so the example can count external effects across a restart.
FIRST_PROGRAM = (
    "from pathlib import Path\n"
    "source = Path('source.txt').read_text(encoding='utf-8')\n"
    "Path('result.txt').write_text(source.strip().upper(), encoding='utf-8')\n"
    "with Path('runs.log').open('a', encoding='utf-8') as log:\n"
    "    log.write('run\\n')\n"
)
REPAIRED_PROGRAM = FIRST_PROGRAM.replace(
    "source.strip().upper(), encoding", "source.strip().upper() + '\\n', encoding"
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_contract_draft() -> WorkContractDraft:
    """One immutable, application-owned definition of done for the transform."""

    return WorkContractDraft(
        contract_id="transform-file",
        version=1,
        objective=(
            "Write result.txt containing the upper-cased source text followed by exactly "
            "one newline."
        ),
        criteria=(
            WorkCriterion(
                criterion_id="content",
                ordinal=1,
                description="result.txt holds the upper-cased task input.",
                evidence_requirement_ids=("artifact",),
            ),
            WorkCriterion(
                criterion_id="format",
                ordinal=2,
                description="result.txt ends with exactly one newline.",
                evidence_requirement_ids=("artifact",),
            ),
        ),
        constraints=(
            WorkConstraint(
                constraint_id="source-unmodified",
                description="The worker leaves its copy of the task input unchanged.",
                evidence_requirement_ids=("source",),
            ),
        ),
        evidence_requirements=(
            WorkEvidenceRequirement(
                requirement_id="artifact",
                kind="workspace.file",
                description="The proposed result.txt version, bound by its content digest.",
            ),
            WorkEvidenceRequirement(
                requirement_id="source",
                kind="workspace.input",
                description="The workspace copy of the task input, bound by its digest.",
            ),
        ),
        verifier=CompletionVerifierRef(
            verifier_id="exact-artifact",
            version="v1",
            configuration_fingerprint=_sha256("exact-artifact:v1"),
        ),
        result_resolver=CompletionResultResolverRef(
            resolver_id="artifact-result",
            version="v1",
            configuration_fingerprint=_sha256("artifact-result:v1"),
        ),
        # Each rejection may continue, but the same gaps twice stop the task.
        continuation_policy=CompletionContinuationPolicy(
            rejection_action=CompletionRejectionAction.CONTINUE,
            max_attempts=3,
            max_repeated_gap_count=1,
        ),
    )


class SessionWorkspaces(EnvironmentFactory):
    """One native workspace per session, addressable again after a restart."""

    def __init__(self, base_root: Path) -> None:
        self.base_root = base_root

    def root_for(self, session_id: str) -> Path:
        # Worker-assigned session ids contain separators; hash them into a portable name.
        return self.base_root / _sha256(session_id)[:16]

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self.root_for(request.session_id)
        root.mkdir(parents=True, exist_ok=True)
        source = request.metadata.get("source_text")
        if isinstance(source, str) and not (root / SOURCE_FILE).exists():
            (root / SOURCE_FILE).write_text(source, encoding="utf-8")
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root),
                runner=LocalRunner(root, inherit_env=False),
                binding=NativeBinding(),
            )
        )

    async def read(self, session_id: str, name: str) -> str | None:
        """Read one worker-written file, or None when it is absent or unusable.

        The library read runs off the event loop, accepts only regular files, and
        stops at a byte limit, so a pipe or a huge file cannot stall the verifier.
        """

        try:
            result = await LocalWorkspace(self.root_for(session_id)).read_bytes(
                name, max_bytes=MAX_FILE_BYTES
            )
        except (OSError, ValueError):
            return None
        if result.truncated:
            return None
        try:
            return result.content.decode("utf-8")
        except UnicodeDecodeError:
            return None

    async def effect_count(self, session_id: str) -> int:
        log = await self.read(session_id, "runs.log")
        return len(log.splitlines()) if log is not None else 0


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ModelStreamEvent:
    return ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)


def _run_program(call_id: str) -> ModelStreamEvent:
    return _tool_call(
        call_id,
        "exec_command",
        {"kind": "process", "argv": [sys.executable, PROGRAM], "timeout_s": 30},
    )


class GapAwareProvider(ModelProvider):
    """Deterministic model that acts on the verifier's continuation gaps."""

    name = "verified-file-script"

    def __init__(
        self, first_program: str = FIRST_PROGRAM, repaired_program: str = REPAIRED_PROGRAM
    ) -> None:
        self.first_program = first_program
        self.repaired_program = repaired_program
        self.requests: list[ModelRequest] = []
        self.received_continuations: list[dict[str, Any]] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        # A stable identity lets a fresh process prove it runs the same model profile.
        return ExecutionProfileBehaviorIdentity(
            name="examples:durable-file-workflow:gap-aware",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        last = request.messages[-1]
        if last.role == "tool":
            yield ModelStreamEvent.text_delta("Produced result.txt for verification.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        text = "".join(part.text for part in last.content if isinstance(part, TextPart))
        if CONTINUATION_TYPE in text:
            continuation = json.loads(text)
            self.received_continuations.append(continuation)
            if [gap["code"] for gap in continuation["gaps"]] != [MISSING_NEWLINE]:
                # This script only knows the newline repair. It stops without changes,
                # so the verifier sees the same gaps and the contract's ceiling decides.
                yield ModelStreamEvent.text_delta("I cannot repair these gaps.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            calls = (
                _tool_call(
                    "write-repair",
                    "write_file",
                    {
                        "path": PROGRAM,
                        "content": self.repaired_program,
                        "mode": "overwrite",
                        "expected_revision": "sha256:" + _sha256(self.first_program),
                    },
                ),
                _run_program("run-repair"),
            )
        else:
            calls = (
                _tool_call(
                    "write-first",
                    "write_file",
                    {"path": PROGRAM, "content": self.first_program, "mode": "create"},
                ),
                _run_program("run-first"),
            )
        for event in calls:
            yield event
        yield _tool_call(f"read-{len(self.requests)}", "read_file", {"path": ARTIFACT})
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


def _evidence(requirement_id: str, content: str | None) -> WorkEvidenceReference:
    kind, reference_id = {
        "artifact": ("workspace.file", ARTIFACT),
        "source": ("workspace.input", SOURCE_FILE),
    }[requirement_id]
    if content is None:
        return WorkEvidenceReference(
            kind=kind,
            reference_id=reference_id,
            requirement_id=requirement_id,
            available=False,
            unavailable_reason=f"{requirement_id}.missing",
        )
    return WorkEvidenceReference(
        kind=kind,
        reference_id=reference_id,
        requirement_id=requirement_id,
        digest=_sha256(content),
    )


def _artifact_result(content: str | None) -> dict[str, object]:
    return {
        "artifact": ARTIFACT,
        "content": content,
        "sha256": None if content is None else _sha256(content),
    }


def artifact_result_reference(content: str | None) -> CompletionResultReference:
    """The result a proposal claims for this artifact; the verifier recomputes it."""

    return CompletionResultReference(
        kind="workspace.file",
        reference_id=ARTIFACT,
        digest=completion_result_sha256(_artifact_result(content)),
    )


class FileArtifactHandler(VerifiedTaskHandler):
    """Start the agent and turn its files into a claim; never decide success here."""

    def __init__(self, workspaces: SessionWorkspaces, limits: RunLimits | None = None) -> None:
        self.workspaces = workspaces
        # Frozen into the task's first attempt and reused by every continuation.
        self.limits = limits or RunLimits()
        self.preparations = 0
        self.proposals = 0

    async def prepare(self, context: VerifiedTaskPreparationContext) -> RunRequest:
        self.preparations += 1
        source = context.task.input.get("source_text")
        if not isinstance(source, str):
            # Raising here holds the task as work_contract_preparation_failed before
            # any model or tool runs.
            raise ValueError("The task has no source_text input.")
        return RunRequest(
            agent_name=AGENT_NAME,
            metadata={"source_text": source},
            messages=[Message.text("user", f"{context.contract.objective}\nSource: {source}")],
            limits=self.limits,
        )

    async def propose(self, context: VerifiedTaskProposalContext) -> VerifiedTaskHandlerReport:
        self.proposals += 1
        session_id = context.attempt.session_id
        # Missing files are proposed as unavailable evidence, never as empty files.
        artifact = await self.workspaces.read(session_id, ARTIFACT)
        return VerifiedTaskHandlerReport(
            proposal=CompletionProposalCreate(
                proposal_id=context.proposal_id,
                attempt_id=context.attempt.attempt_id,
                result=artifact_result_reference(artifact),
                evidence_references=(
                    _evidence("artifact", artifact),
                    _evidence("source", await self.workspaces.read(session_id, SOURCE_FILE)),
                ),
            )
        )


def judge_artifact(
    *,
    artifact: str | None,
    workspace_source: str | None,
    expected_source: str | None,
    proposed: Mapping[str | None, WorkEvidenceReference],
    proposed_result: CompletionResultReference,
) -> CompletionVerifierDecision:
    """Judge each criterion and the constraint independently.

    ``expected_source`` must come from application-owned state. The worker can
    rewrite anything in its workspace, so the workspace copy is evidence to check,
    never the reference to check against.
    """

    artifact_evidence = _evidence("artifact", artifact)
    source_evidence = _evidence("source", workspace_source)
    # Only the exact versions the worker proposed can be judged.
    artifact_ok = artifact is not None and proposed.get("artifact") == artifact_evidence
    # Accepting means accepting the claimed result too, so it must be the one judged.
    result_ok = artifact is not None and proposed_result == artifact_result_reference(artifact)
    source_ok = workspace_source is not None and proposed.get("source") == source_evidence

    # Each subject maps to None when satisfied, or to its (gap code, summary).
    failures: dict[str, tuple[str, str] | None] = {}
    if expected_source is None:
        missing_input = (INPUT_UNAVAILABLE, "The task input is not available to the verifier.")
        failures = dict.fromkeys(("content", "format", "source-unmodified"), missing_input)
    else:
        if not artifact_ok or artifact is None:
            missing = (ARTIFACT_UNAVAILABLE, "The proposed result.txt version is not on disk.")
            failures["content"] = failures["format"] = missing
        elif not result_ok:
            mismatch = (RESULT_MISMATCH, "The proposed result does not match result.txt.")
            failures["content"] = failures["format"] = mismatch
        else:
            body = artifact.rstrip("\n")
            failures["content"] = (
                None
                if body == expected_source.strip().upper()
                else (CONTENT_MISMATCH, "result.txt must hold the upper-cased task input.")
            )
            trailing_newlines = len(artifact) - len(body)
            if trailing_newlines == 1:
                failures["format"] = None
            elif trailing_newlines == 0:
                failures["format"] = (MISSING_NEWLINE, "result.txt must end with one newline.")
            else:
                failures["format"] = (EXTRA_NEWLINE, "result.txt must end with one newline only.")
        if not source_ok:
            failures["source-unmodified"] = (
                SOURCE_UNAVAILABLE,
                "The proposed source.txt version is not on disk.",
            )
        elif workspace_source != expected_source:
            failures["source-unmodified"] = (INPUT_MODIFIED, "The worker changed source.txt.")
        else:
            failures["source-unmodified"] = None

    def judged(
        subject: str,
    ) -> tuple[CriterionOutcomeStatus, str, tuple[WorkEvidenceReference, ...]]:
        evidence = (source_evidence,) if subject == "source-unmodified" else (artifact_evidence,)
        failure = failures[subject]
        if failure is None:
            return CriterionOutcomeStatus.SATISFIED, f"{subject}.verified", evidence
        if failure[0] in {
            ARTIFACT_UNAVAILABLE,
            INPUT_UNAVAILABLE,
            SOURCE_UNAVAILABLE,
            RESULT_MISMATCH,
        }:
            # Nothing trustworthy was read, so nothing is cited.
            return CriterionOutcomeStatus.UNVERIFIABLE, failure[0], ()
        return CriterionOutcomeStatus.UNSATISFIED, failure[0], evidence

    criterion_outcomes = []
    for criterion_id in ("content", "format"):
        status, code, evidence = judged(criterion_id)
        criterion_outcomes.append(
            CompletionCriterionOutcome(
                criterion_id=criterion_id,
                status=status,
                reason_code=code,
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE
                if status is CriterionOutcomeStatus.SATISFIED
                else None,
                evidence_references=evidence,
            )
        )
    status, code, evidence = judged("source-unmodified")
    constraint_outcome = CompletionConstraintOutcome(
        constraint_id="source-unmodified",
        status=status,
        reason_code=code,
        satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE
        if status is CriterionOutcomeStatus.SATISFIED
        else None,
        evidence_references=evidence,
    )
    gaps = [
        CompletionGap(
            criterion_id=subject if subject != "source-unmodified" else None,
            constraint_id=subject if subject == "source-unmodified" else None,
            code=failure[0],
            # A missing task input is the application's gap, not the worker's evidence.
            evidence_requirement_ids=()
            if expected_source is None
            else ("source",)
            if subject == "source-unmodified"
            else ("artifact",),
            summary=failure[1],
        )
        for subject, failure in failures.items()
        if failure is not None
    ]
    # Cayu requires gaps in canonical order: criteria before constraints, then by
    # subject id, code, and evidence requirement ids.
    gaps.sort(
        key=lambda gap: (
            gap.criterion_id is None,
            gap.criterion_id or gap.constraint_id or "",
            gap.code,
            gap.evidence_requirement_ids,
        )
    )
    cited = tuple(
        item for item, ok in ((artifact_evidence, artifact_ok), (source_evidence, source_ok)) if ok
    )
    if expected_source is None:
        # Without the task input nothing can be judged, and no retry by the worker
        # can supply it, so the task is blocked rather than sent another attempt.
        verdict = CompletionVerdict.BLOCKED
    else:
        verdict = CompletionVerdict.REJECTED if gaps else CompletionVerdict.ACCEPTED
    return CompletionVerifierDecision(
        verdict=verdict,
        criterion_outcomes=tuple(criterion_outcomes),
        constraint_outcomes=(constraint_outcome,),
        gaps=tuple(gaps),
        evidence_references=cited,
    )


class ExpectedSource(ABC):
    """Where the verifier gets the value it checks against.

    Its identity is declared in the verifier profile, so swapping the source for a
    different one is a detected profile change, not a silent behavior change.
    """

    @property
    @abstractmethod
    def identity(self) -> ExecutionProfileBehaviorIdentity: ...

    @abstractmethod
    async def load(self, request: CompletionVerifierRequest) -> str | None: ...


class TaskInputSource(ExpectedSource):
    """Read the expected input from the task record instead of the workspace.

    This only keeps the expectation out of the worker's reach if the worker's
    execution cannot reach the task store either. The example's local, unsandboxed
    runner can reach it in SQLite mode; production needs a sandboxed runner.
    """

    def __init__(self, tasks: InMemoryTaskStore | SQLiteTaskStore) -> None:
        self.tasks = tasks

    @property
    def identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples:durable-file-workflow:task-input-source",
            behavior_version="1",
            implementation_version="1",
        )

    async def load(self, request: CompletionVerifierRequest) -> str | None:
        task = await self.tasks.load_task(request.attempt.task_id)
        source = task.input.get("source_text") if task is not None else None
        return source if isinstance(source, str) else None


class ExactArtifactVerifier(DeterministicCompletionVerifier):
    """Independent, side-effect-free check of the proposed files against the task input."""

    def __init__(self, workspaces: SessionWorkspaces, expected_source: ExpectedSource) -> None:
        self.workspaces = workspaces
        self.expected_source = expected_source
        self.calls = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="examples:durable-file-workflow:exact-artifact",
            behavior_version="1",
            implementation_version="1",
        )

    @property
    def execution_profile_components(
        self,
    ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        # The expected-input source decides acceptance, so the injected source's own
        # identity is part of the verifier profile.
        return (
            CompletionVerifierProfileComponentDeclaration(
                component_id="expected-source", identity=self.expected_source.identity
            ),
        )

    async def verify(self, request: CompletionVerifierRequest) -> CompletionVerifierDecision:
        self.calls += 1
        session_id = request.attempt.session_id
        return judge_artifact(
            artifact=await self.workspaces.read(session_id, ARTIFACT),
            workspace_source=await self.workspaces.read(session_id, SOURCE_FILE),
            expected_source=await self.expected_source.load(request),
            proposed={item.requirement_id: item for item in request.proposal.evidence_references},
            proposed_result=request.proposal.result,
        )


class ArtifactResultResolver(CompletionResultResolver):
    """Rebuild the task result from the accepted artifact; Cayu checks its digest."""

    def __init__(self, workspaces: SessionWorkspaces) -> None:
        self.workspaces = workspaces
        self.calls = 0

    async def resolve(self, request: CompletionResultResolverRequest) -> dict[str, object]:
        self.calls += 1
        content = await self.workspaces.read(request.attempt.session_id, ARTIFACT)
        if content is None:
            raise RuntimeError("Accepted artifact is no longer available.")
        return _artifact_result(content)


def lose_proposal_acknowledgement(app: CayuApp, *, on_proposal: int) -> None:
    """Demo-only fault: the store commits a proposal, but the worker never hears back.

    This stands in for a dropped connection or a crash right after a durable write.
    """

    commit = app.submit_work_attempt_proposal
    committed = 0

    async def commit_then_lose_reply(request: WorkAttemptProposalRequest) -> CompletionProposal:
        nonlocal committed
        proposal = await commit(request)
        committed += 1
        if committed == on_proposal:
            raise ConnectionError(f"Lost the acknowledgement for committed proposal {committed}.")
        return proposal

    # Patching one app instance keeps the fault out of the reusable components.
    app.submit_work_attempt_proposal = commit_then_lose_reply  # ty: ignore[invalid-assignment]


@dataclass
class Components:
    """Parts that outlive one app instance; each built app gets its own verifier."""

    workspaces: SessionWorkspaces
    provider: GapAwareProvider
    handler: FileArtifactHandler
    resolver: ArtifactResultResolver
    verifiers: list[ExactArtifactVerifier] = field(default_factory=list)


def build_app(
    sessions: InMemorySessionStore | SQLiteSessionStore,
    tasks: InMemoryTaskStore | SQLiteTaskStore,
    parts: Components,
) -> CayuApp:
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    app.register_provider(parts.provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="session-workspaces"), parts.workspaces, default=True
    )
    register_file_worker(app, workspace_root=parts.workspaces.base_root, system_prompt=GOAL_PROMPT)
    # Register the exact verifier and resolver the contract freezes, before any worker runs.
    draft = build_contract_draft()
    verifier = ExactArtifactVerifier(parts.workspaces, TaskInputSource(tasks))
    parts.verifiers.append(verifier)
    app.register_completion_verifier(draft.verifier, verifier)
    app.register_completion_result_resolver(draft.result_resolver, parts.resolver)
    return app


@dataclass(frozen=True)
class AttemptRecord:
    """Durable evidence for one attempt, read back through public store queries."""

    ordinal: int
    kind: str
    run_by: str
    proposal: CompletionProposal | None
    decision: CompletionDecision | None
    application: CompletionDecisionApplicationReceipt | None

    @property
    def verdict(self) -> str:
        return self.decision.verdict.value if self.decision else "-"

    @property
    def gap_codes(self) -> tuple[str, ...]:
        return tuple(gap.code for gap in self.decision.gaps) if self.decision else ()


@dataclass(frozen=True)
class VerifiedDemoResult:
    task: Task
    attempts: tuple[AttemptRecord, ...]
    attempts_before_restart: tuple[AttemptRecord, ...]
    replayed_task: Task | None
    interrupted_by: str | None
    session_id: str
    binding_retired: bool
    active_contract_task: Task | None
    unsettled_admissions: int
    resolved_events: int
    received_continuations: tuple[dict[str, Any], ...]
    provider_calls_before_restart: int
    provider_calls: int
    effects_before_restart: int
    effects: int
    verifier_calls: int
    resolver_calls: int
    preparations: int
    proposals: int


def demo_alias_codec() -> PublicAuthorityAliasCodec:
    """A throwaway key for this run's durable workspace-observation aliases.

    Deployments load a stable keyring with public_authority_alias_codec_from_environment()
    so a restarted process can verify aliases written before it.
    """

    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="demo", keys={"demo": SecretStr(key)})
    )


def _open_stores(
    store: str, state: Path, alias_codec: PublicAuthorityAliasCodec
) -> tuple[InMemorySessionStore | SQLiteSessionStore, InMemoryTaskStore | SQLiteTaskStore]:
    if store == "memory":
        return InMemorySessionStore(), InMemoryTaskStore()
    state.mkdir(parents=True, exist_ok=True)
    return (
        SQLiteSessionStore(state / "sessions.sqlite", public_authority_alias_codec=alias_codec),
        SQLiteTaskStore(state / "tasks.sqlite"),
    )


async def _close_stores(
    sessions: InMemorySessionStore | SQLiteSessionStore,
    tasks: InMemoryTaskStore | SQLiteTaskStore,
) -> None:
    if isinstance(tasks, SQLiteTaskStore):
        await tasks.close()
    if isinstance(sessions, SQLiteSessionStore):
        await sessions.close()


CLOSE_ATTEMPTS = 3
# Generous: CI runners are slow, and this bounds the demo, not the contract.
RUN_TIMEOUT_SECONDS = 300


def _still_draining(error: BaseException) -> bool:
    """Whether the worker reported VerifiedTaskWorkerDraining.

    The report can arrive alone, inside an exception group, or as the cause of a
    cancellation, which Cayu keeps outermost.
    """

    if isinstance(error, VerifiedTaskWorkerDraining):
        return True
    if isinstance(error, BaseExceptionGroup) and any(
        _still_draining(item) for item in error.exceptions
    ):
        return True
    cause = error.__cause__
    return (
        isinstance(error, asyncio.CancelledError) and cause is not None and _still_draining(cause)
    )


async def _finish_draining(worker: VerifiedTaskWorker) -> None:
    """Retry closing a worker that still owns in-flight work.

    The worker contract says to keep a draining worker and its stores and retry.
    If it never finishes, the last draining report propagates and the caller must
    not close the stores.
    """

    for attempt in range(1, CLOSE_ATTEMPTS + 1):
        try:
            await worker.aclose()
            return
        except BaseException as error:
            if attempt == CLOSE_ATTEMPTS or not _still_draining(error):
                raise


async def _run_worker(app: CayuApp, handler: FileArtifactHandler, worker_id: str) -> None:
    worker = VerifiedTaskWorker(app, handler, worker_id=worker_id, poll_interval_s=0.01)
    try:
        async with worker:
            await asyncio.wait_for(worker.run(max_tasks=1), RUN_TIMEOUT_SECONDS)
    except BaseException as error:
        if not _still_draining(error):
            raise
        try:
            await _finish_draining(worker)
        except BaseException as close_error:
            # Keep a cancellation outermost, carrying the unfinished drain as its
            # cause so callers still see that work remains in flight.
            if isinstance(error, asyncio.CancelledError):
                raise error from close_error
            raise
        # The worker has now closed. A cancellation still propagates, without the
        # drain it carried, since that work has settled; otherwise report only what
        # the run itself raised.
        if isinstance(error, asyncio.CancelledError):
            cause = error.__cause__
            if isinstance(cause, BaseExceptionGroup):
                _, cause = cause.split(VerifiedTaskWorkerDraining)
            elif isinstance(cause, VerifiedTaskWorkerDraining):
                cause = None
            raise error from cause
        if isinstance(error, BaseExceptionGroup):
            _, remaining = error.split(VerifiedTaskWorkerDraining)
            if remaining is not None:
                # Re-raise unchanged, keeping each failure's own cause.
                if len(remaining.exceptions) == 1:
                    raise remaining.exceptions[0]  # noqa: B904
                raise remaining  # noqa: B904


async def _attempt_history(tasks: InMemoryTaskStore | SQLiteTaskStore) -> tuple[AttemptRecord, ...]:
    """Walk the durable attempt chain backwards from the latest admission.

    Each decision's application receipt is keyed by the idempotency key that the
    next step records: the successor's continuation, or the final lifecycle receipt.
    """

    records: list[AttemptRecord] = []
    admission = await tasks.load_latest_work_attempt_admission(TASK_ID)
    lifecycle = (
        await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
        if admission is not None
        else None
    )
    application_key = lifecycle.request.application_idempotency_key if lifecycle else None
    while admission is not None:
        if admission.attempt is None:
            raise RuntimeError("An admitted attempt has no durable attempt record.")
        proposal = await tasks.load_completion_proposal_for_attempt(admission.attempt_id)
        decision = (
            await tasks.load_completion_decision_for_proposal(proposal.proposal_id)
            if proposal is not None
            else None
        )
        application = (
            await tasks.load_completion_decision_application_receipt(TASK_ID, application_key)
            if application_key is not None
            else None
        )
        records.append(
            AttemptRecord(
                ordinal=admission.attempt.ordinal,
                kind=admission.kind,
                run_by=admission.claim.worker_id,
                proposal=proposal,
                decision=decision,
                application=application,
            )
        )
        previous = admission.continuation
        application_key = previous.application_idempotency_key if previous else None
        admission = (
            await tasks.load_work_attempt_admission(previous.prior_admission_id)
            if previous is not None
            else None
        )
    return tuple(reversed(records))


async def run_demo(
    base_root: Path,
    *,
    store: str = "memory",
    lose_acknowledgement: bool = True,
    provider: GapAwareProvider | None = None,
    limits: RunLimits | None = None,
) -> VerifiedDemoResult:
    alias_codec = demo_alias_codec()
    sessions, tasks = _open_stores(store, base_root / "state", alias_codec)
    workspaces = SessionWorkspaces(base_root / "workspaces")
    parts = Components(
        workspaces=workspaces,
        provider=provider or GapAwareProvider(),
        handler=FileArtifactHandler(workspaces, limits),
        resolver=ArtifactResultResolver(workspaces),
    )
    keep_stores_open = False
    try:
        app = build_app(sessions, tasks, parts)
        if lose_acknowledgement:
            lose_proposal_acknowledgement(app, on_proposal=2)
        contract = await app.create_work_contract(build_contract_draft())
        await app.create_task(
            TaskCreate(
                task_id=TASK_ID,
                type="transform_file",
                input={"source_text": SOURCE_TEXT},
                work_contract=contract.reference(),
            )
        )
        interrupted_by: str | None = None
        try:
            await _run_worker(app, parts.handler, "worker-before-restart")
        except ConnectionError as error:
            interrupted_by = str(error)
        admission = await tasks.load_latest_work_attempt_admission(TASK_ID)
        if admission is None:
            raise RuntimeError("The worker never admitted an attempt.")
        session_id = admission.session_id
        attempts_before_restart = await _attempt_history(tasks)
        provider_calls_before_restart = len(parts.provider.requests)
        effects_before_restart = await parts.workspaces.effect_count(session_id)

        if interrupted_by is not None:
            # Simulate a restart: a fresh app, and for SQLite freshly opened stores,
            # recovering the same task from durable evidence alone.
            if store != "memory":
                await _close_stores(sessions, tasks)
                sessions, tasks = _open_stores(store, base_root / "state", alias_codec)
            app = build_app(sessions, tasks, parts)
            await _run_worker(app, parts.handler, "worker-after-restart")

        task = await tasks.load_task(TASK_ID)
        latest = await tasks.load_latest_work_attempt_admission(TASK_ID)
        if task is None or latest is None:
            raise RuntimeError("Demo task disappeared.")
        attempts = await _attempt_history(tasks)
        accepted = attempts[-1].application if attempts[-1].verdict == "accepted" else None
        # Asking again with the same idempotency key replays the stored outcome; it
        # neither calls the resolver nor applies a second decision.
        replayed_task = (
            await app.resolve_completion_result(
                CompletionResultResolutionRequest(
                    task_id=TASK_ID,
                    decision_id=accepted.decision_id,
                    idempotency_key=accepted.idempotency_key,
                )
            )
            if accepted is not None
            else None
        )
        receipt = await tasks.load_work_attempt_lifecycle_receipt(latest.admission_id)
        resolved = await sessions.query_events(
            EventQuery(session_id=session_id, event_type=EventType.TASK_COMPLETION_RESULT_RESOLVED)
        )
        return VerifiedDemoResult(
            task=task,
            attempts=attempts,
            attempts_before_restart=attempts_before_restart,
            replayed_task=replayed_task,
            interrupted_by=interrupted_by,
            session_id=session_id,
            binding_retired=receipt is not None and receipt.retired_contract_binding,
            active_contract_task=await tasks.load_active_work_contract_task_for_session(session_id),
            unsettled_admissions=len(await tasks.list_unsettled_work_attempt_admissions()),
            resolved_events=len(resolved),
            received_continuations=tuple(parts.provider.received_continuations),
            provider_calls_before_restart=provider_calls_before_restart,
            provider_calls=len(parts.provider.requests),
            effects_before_restart=effects_before_restart,
            effects=await parts.workspaces.effect_count(session_id),
            verifier_calls=sum(verifier.calls for verifier in parts.verifiers),
            resolver_calls=parts.resolver.calls,
            preparations=parts.handler.preparations,
            proposals=parts.handler.proposals,
        )
    except BaseException as error:
        # A worker that never finished draining still owns in-flight work, so its
        # stores must stay open; closing them could strand that work.
        keep_stores_open = _still_draining(error)
        raise
    finally:
        if not keep_stores_open:
            await _close_stores(sessions, tasks)


def durable_evidence_preserved(
    before: tuple[AttemptRecord, ...], after: tuple[AttemptRecord, ...]
) -> bool:
    """Every record that existed before the restart is still equal afterwards."""

    if len(after) < len(before):
        return False
    return all(
        earlier is None or earlier == later
        for old, new in zip(before, after, strict=False)
        for earlier, later in (
            (old.proposal, new.proposal),
            (old.decision, new.decision),
            (old.application, new.application),
        )
    )


def print_timeline(result: VerifiedDemoResult) -> None:
    for attempt in result.attempts:
        gaps = f" gaps={','.join(attempt.gap_codes)}" if attempt.gap_codes else ""
        print(f"attempt {attempt.ordinal} ({attempt.kind}) run by {attempt.run_by}")
        if attempt.proposal:
            print(f"  proposal  {attempt.proposal.proposal_id}")
        if attempt.decision:
            verifier = attempt.decision.verifier
            print(
                f"  decision  {attempt.decision.decision_id} {attempt.verdict}{gaps} "
                f"by {verifier.verifier_id}@{verifier.version} via {attempt.decision.worker_id}"
            )
        if attempt.application:
            print(
                f"  applied   {attempt.application.idempotency_key} -> "
                f"task {attempt.application.task.status.value}"
            )
    if result.interrupted_by:
        same = durable_evidence_preserved(result.attempts_before_restart, result.attempts)
        print(f"worker interrupted: {result.interrupted_by}")
        print(
            f"restart reused durable evidence unchanged={same}: "
            f"model calls {result.provider_calls_before_restart} -> {result.provider_calls}, "
            f"program runs {result.effects_before_restart} -> {result.effects}"
        )
    print(f"task {result.task.id}: {result.task.status.value} result={result.task.result}")
    print(
        f"verifier calls={result.verifier_calls} resolver calls={result.resolver_calls} "
        f"resolved events={result.resolved_events} "
        f"replay unchanged={result.replayed_task == result.task}"
    )
    print(
        f"contract binding retired={result.binding_retired} "
        f"session released={result.active_contract_task is None}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the contract-verified file workflow.")
    parser.add_argument("--store", choices=("memory", "sqlite"), default="memory")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="cayu-verified-file-") as temp:
        print_timeline(await run_demo(Path(temp), store=args.store))


if __name__ == "__main__":
    asyncio.run(main())
