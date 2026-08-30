"""Durable execution of authority-free scenario-v2 stimuli."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NoReturn

from cayu.artifacts import (
    FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES,
    FILE_ATTACHMENT_IMAGE_CONTENT_TYPES,
    FileAttachment,
    FileAttachmentKind,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import FilePart, Message, MessageRole, TextPart
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.capacity import EvalExecutionCapacity
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationSourceIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
)
from cayu.evals.execution import (
    CompiledCorpusSuite,
    CorpusExecutionResult,
    CorpusTarget,
    _finalize_compiled_corpus_result,
    evaluation_target_identity,
)
from cayu.evals.external import (
    ExternalTrialEnvelopeV1,
    ExternalTrialIdentityV1,
)
from cayu.evals.memory_attribution import (
    eval_memory_attribution_bounds_for_trial_count,
    eval_memory_attribution_max_bytes_for_trial_count,
    eval_memory_attribution_source_limit_for_trial_count,
)
from cayu.evals.models import EvalRun, aggregate_eval_score, aggregate_eval_status
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES,
)
from cayu.evals.runner import (
    _aggregate_trials,
    _FreshMemoryAttributionReadLifecycle,
    _run_case_once_with_public_projection,
)
from cayu.evals.scenario import (
    EvalScenarioDocumentV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInputV2,
    ScenarioJsonPartV2,
    ScenarioQueuedInputEventV2,
    ScenarioResumedInputEventV2,
    ScenarioTextPartV2,
    compile_eval_scenario,
)
from cayu.evals.scenario_preflight import (
    ScenarioLaunchBindingV2,
    ScenarioLaunchSettingsV2,
)
from cayu.evals.store import (
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunCostBudget,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalScenarioRunInvocation,
    EvalScenarioRunProgress,
    EvalScenarioTrialFailureCode,
    EvalScenarioTrialPhase,
    EvalScenarioTrialProgress,
    EvalStore,
)
from cayu.runtime.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    ExecutionProfileMismatchError,
)
from cayu.runtime.sessions import (
    EnqueueSessionMessageRequest,
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    ResumeRequest,
    RunRequest,
    SessionMessageDeliveryMode,
    SessionStatus,
    copy_run_request,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.user_input import UserInputResponse


class ScenarioExecutionError(RuntimeError):
    """Safe orchestration failure retained in bounded scenario progress."""

    def __init__(self, code: EvalScenarioTrialFailureCode, message: str) -> None:
        self.code = EvalScenarioTrialFailureCode(code)
        super().__init__(message)


def scenario_launch_settings_from_invocation(
    invocation: EvalScenarioRunInvocation,
    *,
    max_concurrency: int,
    max_steps: int | None,
    limits: RunLimits | None,
    cost_budget: EvalRunCostBudget | None,
) -> ScenarioLaunchSettingsV2:
    """Reconstruct the exact public launch selection retained at admission."""

    if type(invocation) is not EvalScenarioRunInvocation:
        raise TypeError("invocation must be an exact EvalScenarioRunInvocation.")
    return ScenarioLaunchSettingsV2(
        environment_name=invocation.environment_name,
        trials=invocation.trials,
        max_concurrency=max_concurrency,
        timeout_seconds=invocation.timeout_seconds,
        max_steps=max_steps,
        limits=limits,
        cost_budget=cost_budget,
        artifact_references={
            item.requirement_id: item.artifact_id for item in invocation.artifact_references
        },
    )


def corpus_for_eval_scenario(
    scenario: EvalScenarioDocumentV2,
    binding: ScenarioLaunchBindingV2,
    target: CorpusTarget,
    *,
    project_root: Path | None = None,
) -> EvalCorpusDocument:
    """Create the immutable one-case result contract used by a scenario run."""

    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    if type(binding) is not ScenarioLaunchBindingV2:
        raise TypeError("binding must be an exact ScenarioLaunchBindingV2.")
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    target_identity = evaluation_target_identity(target, project_root=project_root)
    if (
        scenario.revision != binding.scenario_revision
        or scenario.target_key != binding.target_key
        or target_identity.target_key != binding.target_key
        or target_identity.application_release_id != binding.application_release_id
        or target_identity.app_manifest_fingerprint != binding.app_manifest_fingerprint
    ):
        raise ValueError("Scenario launch binding does not match the current target.")
    source = scenario.source or EvaluationSourceIdentityV1(
        application_release_id=target_identity.application_release_id,
        app_manifest_schema_version=target_identity.app_manifest_schema_version,
        app_manifest_fingerprint=target_identity.app_manifest_fingerprint,
        evidence_revision=scenario.revision,
    )
    suite = EvalSuiteSpec.create(
        id="scenario",
        name=f"Scenario: {scenario.name}",
        description=scenario.description,
        trial_request=TrialRequestSpec(
            trials=binding.trials,
            timeout_seconds=binding.timeout_seconds,
        ),
    )
    case = EvalCaseSpec.create(
        id=scenario.id,
        suite_id=suite.id,
        name=scenario.name,
        description=scenario.description,
        source=source,
        # Corpus v1 remains text-only. The worker replaces this inert marker
        # with the exact typed scenario input before runtime admission.
        input=RunInputSpec(
            messages=(
                CorpusUserMessageSpec(text=f"Execute controlled scenario {scenario.revision}."),
            )
        ),
        assertions=(
            RootStatusAssertionSpec(
                id="scenario-completed",
                expected="completed",
                description="The controlled scenario reaches a completed root session.",
            ),
        ),
    )
    return EvalCorpusDocument.create(
        target_key=scenario.target_key,
        evidence_policy=target.evidence_policy,
        suites=(suite,),
        cases=(case,),
    )


def _artifact_maps(
    scenario: EvalScenarioDocumentV2,
    binding: ScenarioLaunchBindingV2,
) -> tuple[dict[str, ScenarioArtifactRequirementV2], dict[str, str]]:
    return (
        {item.id: item for item in scenario.artifact_requirements},
        {item.requirement_id: item.artifact_id for item in binding.artifacts},
    )


def _scenario_messages(
    scenario_input: ScenarioInputV2,
    scenario: EvalScenarioDocumentV2,
    binding: ScenarioLaunchBindingV2,
) -> list[Message]:
    requirements, artifact_ids = _artifact_maps(scenario, binding)
    messages: list[Message] = []
    for authored in scenario_input.messages:
        content: list[TextPart | FilePart] = []
        for part in authored.content:
            if type(part) is ScenarioTextPartV2:
                content.append(TextPart(text=part.text))
            elif type(part) is ScenarioJsonPartV2:
                content.append(
                    TextPart(
                        text=json.dumps(
                            part.model_dump(mode="json")["value"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                )
            elif type(part) is ScenarioFilePartV2:
                requirement = requirements[part.artifact_requirement_id]
                artifact_id = artifact_ids[part.artifact_requirement_id]
                if requirement.content_type in FILE_ATTACHMENT_IMAGE_CONTENT_TYPES:
                    kind = FileAttachmentKind.IMAGE
                elif requirement.content_type in FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES:
                    kind = FileAttachmentKind.DOCUMENT
                else:
                    raise ScenarioExecutionError(
                        EvalScenarioTrialFailureCode.EXECUTION_FAILED,
                        "Scenario file content type is not supported by runtime attachments.",
                    )
                attachment = FileAttachment(
                    artifact_id=artifact_id,
                    kind=kind,
                    filename=requirement.filename,
                    content_type=requirement.content_type,
                    size_bytes=requirement.size_bytes,
                )
                content.append(FilePart(attachment=attachment.model_dump(mode="json")))
            else:  # pragma: no cover - the closed scenario union is validated earlier
                raise TypeError("Unsupported scenario input part.")
        messages.append(Message(role=MessageRole.USER, content=tuple(content)))
    return messages


def _queued_message_text(message: Message) -> str:
    rendered: list[str] = []
    for part in message.content:
        if type(part) is TextPart:
            rendered.append(part.text)
        elif type(part) is FilePart:
            attachment = FileAttachment.model_validate(part.attachment)
            rendered.append(f"Attached file: {attachment.filename}")
    return "\n".join(rendered)


def _resumed_answer_text(message: Message) -> str:
    rendered: list[str] = []
    for part in message.content:
        if type(part) is TextPart:
            rendered.append(part.text)
        elif type(part) is FilePart:
            attachment = FileAttachment.model_validate(part.attachment)
            rendered.append(f"Attached file: {attachment.filename}")
    return "\n".join(rendered)


def _resolution_actor(actor_id: str) -> ResolutionActor:
    return ResolutionActor(
        subject=actor_id,
        source=(
            ResolutionActorSource.SYSTEM
            if actor_id.startswith("cayu:")
            else ResolutionActorSource.HTTP_AUTH
        ),
    )


async def _pending_action(
    target: CorpusTarget,
    session_id: str,
    kind: PendingActionKind,
) -> PendingActionRecord:
    result = await target.app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id, kind=kind, limit=16)
    )
    if len(result.actions) != 1:
        code = (
            EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE
            if kind is PendingActionKind.TOOL_APPROVAL
            else EvalScenarioTrialFailureCode.EXPECTED_USER_INPUT_UNAVAILABLE
        )
        raise ScenarioExecutionError(code, f"Expected one current {kind.value} action.")
    return result.actions[0]


async def _approval_occurrence(
    target: CorpusTarget,
    session_id: str,
    tool_name: str,
) -> int:
    """Count distinct durable approval requests for one tool through the current pause."""

    events = await target.app.session_store.load_events(session_id)
    approval_ids: set[str] = set()
    for event in events:
        if event.type is not EventType.TOOL_CALL_APPROVAL_REQUESTED:
            continue
        if event.tool_name != tool_name:
            continue
        approval_id = event.payload.get("approval_id")
        if type(approval_id) is not str or not approval_id:
            raise ScenarioExecutionError(
                EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE,
                "Approval history is missing its durable request identity.",
            )
        approval_ids.add(approval_id)
    return len(approval_ids)


class _ScenarioTrialDriver:
    def __init__(
        self,
        *,
        target: CorpusTarget,
        scenario: EvalScenarioDocumentV2,
        binding: ScenarioLaunchBindingV2,
        store: EvalStore,
        claim: EvalRunClaim,
        trial_number: int,
        initial_progress: EvalScenarioTrialProgress,
        poll_seconds: float,
        manifest_project_root: Path | None,
        expected_app_manifest_fingerprint: str | None,
        expected_execution_profile: ExecutionProfileIdentity | None,
        external_envelope: ExternalTrialEnvelopeV1 | None,
    ) -> None:
        self.target = target
        self.scenario = scenario
        self.binding = binding
        self.store = store
        self.claim = claim
        self.trial_number = trial_number
        self.poll_seconds = poll_seconds
        self.manifest_project_root = manifest_project_root
        self.expected_app_manifest_fingerprint = expected_app_manifest_fingerprint
        self.expected_execution_profile = expected_execution_profile
        self.external_envelope = external_envelope
        self.initial_progress = initial_progress.model_copy(deep=True)
        session_id = (
            initial_progress.session_id
            if initial_progress.phase
            in {
                EvalScenarioTrialPhase.AWAITING_APPROVAL,
                EvalScenarioTrialPhase.AWAITING_RESUME,
            }
            else f"{claim.run_id}-scenario-{claim.epoch}-{trial_number}"
        )
        if session_id is None:
            raise ValueError("A resumable scenario checkpoint requires its durable session id.")
        self.session_id: str = session_id
        self.next_sequence = initial_progress.next_event_sequence

    async def _require_expected_execution_profile(self) -> None:
        """Reject profile drift before a scenario continuation can do governed work."""

        if self.expected_app_manifest_fingerprint is not None:
            current_target = evaluation_target_identity(
                self.target,
                project_root=self.manifest_project_root,
            )
            if current_target.app_manifest_fingerprint != self.expected_app_manifest_fingerprint:
                raise EvalExecutionProfileChangedError(
                    "Scenario application identity changed after eval admission.",
                )
        if self.expected_execution_profile is None:
            return
        prepared = await self.target.app._session_engine._prepare_initial_run(
            copy_run_request(self.target.request_base),
            admit_session=False,
        )
        if prepared is None or prepared.execution_profile != self.expected_execution_profile:
            raise EvalExecutionProfileChangedError(
                "Scenario runtime execution profile changed after eval admission.",
            )

    async def _update(
        self,
        phase: EvalScenarioTrialPhase,
        *,
        pending_event_id: str | None = None,
        pending_tool_name: str | None = None,
        pending_input_id: str | None = None,
        pending_resume_kind: Literal["user_input", "manual_recovery"] | None = None,
        failure_code: EvalScenarioTrialFailureCode | None = None,
    ) -> None:
        await self.store.update_scenario_trial(
            self.claim,
            EvalScenarioTrialProgress(
                trial_number=self.trial_number,
                phase=phase,
                session_id=self.session_id,
                next_event_sequence=self.next_sequence,
                pending_event_id=pending_event_id,
                pending_tool_name=pending_tool_name,
                pending_input_id=pending_input_id,
                pending_resume_kind=pending_resume_kind,
                failure_code=failure_code,
            ),
        )

    async def _enter_running_phase(self) -> None:
        """Fence the durable progress write immediately before runtime dispatch."""

        await self._require_expected_execution_profile()
        await self._update(EvalScenarioTrialPhase.RUNNING)
        await self._require_expected_execution_profile()

    async def _enqueue_ready_steps(self) -> None:
        events = self.scenario.events
        while self.next_sequence < len(events):
            event = events[self.next_sequence]
            if type(event) is not ScenarioQueuedInputEventV2:
                return
            messages = _scenario_messages(event.input, self.scenario, self.binding)
            for message_index, message in enumerate(messages):
                await self.target.app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=self.session_id,
                        idempotency_key=(
                            f"{self.claim.run_id}:{self.claim.epoch}:{self.trial_number}:"
                            f"{event.id}:{message_index}"
                        ),
                        content=_queued_message_text(message),
                        message=message,
                        delivery_mode=SessionMessageDeliveryMode(event.delivery_mode),
                        requested_by=ResolutionActor(
                            subject="cayu:eval-scenario",
                            source=ResolutionActorSource.SYSTEM,
                        ),
                    )
                )
            self.next_sequence += 1
            await self._update(EvalScenarioTrialPhase.RUNNING)

    async def _drain(self, stream: AsyncIterator[Event]) -> AsyncIterator[Event]:
        first = True
        async for event in stream:
            if first:
                first = False
                await self._enqueue_ready_steps()
            yield event

    async def _wait_for_approval(
        self,
        event: ScenarioApprovalCheckpointEventV2,
        *,
        checkpoint_persisted: bool = False,
    ) -> AsyncIterator[Event]:
        action = await _pending_action(
            self.target,
            self.session_id,
            PendingActionKind.TOOL_APPROVAL,
        )
        if action.tool_name != event.tool_name:
            raise ScenarioExecutionError(
                EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE,
                "Current approval tool does not match the scenario checkpoint.",
            )
        occurrence = await _approval_occurrence(
            self.target,
            self.session_id,
            event.tool_name,
        )
        if occurrence != event.occurrence:
            raise ScenarioExecutionError(
                EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE,
                "Current approval occurrence does not match the scenario checkpoint.",
            )
        if checkpoint_persisted:
            if (
                self.initial_progress.pending_event_id != event.id
                or self.initial_progress.pending_tool_name != event.tool_name
            ):
                raise ScenarioExecutionError(
                    EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE,
                    "Durable approval progress does not match the scenario checkpoint.",
                )
        else:
            self.next_sequence += 1
            await self._update(
                EvalScenarioTrialPhase.AWAITING_APPROVAL,
                pending_event_id=event.id,
                pending_tool_name=event.tool_name,
            )
        while True:
            run = await self.store.load_run(self.claim.run_id)
            if run is None or run.status is not EvalRunStatus.RUNNING:
                raise EvalRunClaimLost("Scenario run stopped while awaiting approval.")
            progress = run.scenario_progress
            if progress is None or progress.attempt != self.claim.epoch:
                raise EvalRunClaimLost("Scenario progress changed while awaiting approval.")
            current = progress.trials[self.trial_number - 1]
            if current.approval is not None:
                decision = current.approval
                break
            await asyncio.sleep(self.poll_seconds)
        # Re-query after the operator decision. The private runtime linkage must
        # still identify the same live checkpoint; it is never persisted in Evals.
        action = await _pending_action(
            self.target,
            self.session_id,
            PendingActionKind.TOOL_APPROVAL,
        )
        if (
            action.tool_name != event.tool_name
            or action.approval_id is None
            or action.round_id is None
            or action.tool_call_id is None
        ):
            raise ScenarioExecutionError(
                EvalScenarioTrialFailureCode.EXPECTED_APPROVAL_UNAVAILABLE,
                "Current approval linkage no longer matches the scenario checkpoint.",
            )
        await self._enter_running_phase()
        return self.target.app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=self.session_id,
                approval_id=action.approval_id,
                tool_round_id=action.round_id,
                tool_call_id=action.tool_call_id,
                decision=ToolApprovalDecision(decision.decision),
                reason=decision.reason,
                resolved_by=_resolution_actor(decision.actor_id),
            )
        )

    async def _user_input_stream(
        self,
        event: ScenarioResumedInputEventV2,
        *,
        checkpoint_persisted: bool = False,
    ) -> AsyncIterator[Event]:
        action = await _pending_action(
            self.target,
            self.session_id,
            PendingActionKind.USER_INPUT,
        )
        if action.input_id is None:
            raise ScenarioExecutionError(
                EvalScenarioTrialFailureCode.EXPECTED_USER_INPUT_UNAVAILABLE,
                "Current user-input action has no durable runtime linkage.",
            )
        if checkpoint_persisted:
            if (
                self.initial_progress.pending_event_id != event.id
                or self.initial_progress.pending_resume_kind != "user_input"
                or self.initial_progress.pending_input_id != action.input_id
            ):
                raise ScenarioExecutionError(
                    EvalScenarioTrialFailureCode.EXPECTED_USER_INPUT_UNAVAILABLE,
                    "Durable user-input progress does not match the current checkpoint.",
                )
        else:
            self.next_sequence += 1
            await self._update(
                EvalScenarioTrialPhase.AWAITING_RESUME,
                pending_event_id=event.id,
                pending_input_id=action.input_id,
                pending_resume_kind="user_input",
            )
        messages = _scenario_messages(event.input, self.scenario, self.binding)
        answer = "\n\n".join(_resumed_answer_text(message) for message in messages)
        artifacts = [
            part.attachment
            for message in messages
            for part in message.content
            if type(part) is FilePart
        ]
        await self._enter_running_phase()
        return self.target.app.resolve_user_input(
            UserInputResponse(
                session_id=self.session_id,
                input_id=action.input_id,
                answer=answer,
                artifacts=artifacts,
                resolved_by=ResolutionActor(
                    subject="cayu:eval-scenario",
                    source=ResolutionActorSource.SYSTEM,
                ),
            )
        )

    async def _resume_stream(
        self,
        event: ScenarioResumedInputEventV2,
        *,
        checkpoint_persisted: bool = False,
    ) -> AsyncIterator[Event]:
        if checkpoint_persisted:
            if (
                self.initial_progress.pending_event_id != event.id
                or self.initial_progress.pending_resume_kind != "manual_recovery"
                or self.initial_progress.pending_input_id is not None
            ):
                self._unexpected("Durable session-resume progress does not match its event.")
        else:
            self.next_sequence += 1
            await self._update(
                EvalScenarioTrialPhase.AWAITING_RESUME,
                pending_event_id=event.id,
                pending_resume_kind="manual_recovery",
            )
        messages = _scenario_messages(event.input, self.scenario, self.binding)
        request = self.target.request_base
        await self._enter_running_phase()
        return self.target.app.resume(
            ResumeRequest(
                session_id=self.session_id,
                messages=messages,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=request.retry_policy,
                structured_output=request.structured_output,
                thinking=request.thinking,
                loop_policies=request.loop_policies,
            )
        )

    def _unexpected(self, message: str) -> NoReturn:
        raise ScenarioExecutionError(
            EvalScenarioTrialFailureCode.UNEXPECTED_SESSION_STATE,
            message,
        )

    async def reconcile_terminal_progress(self) -> None:
        """Close progress when the generic trial deadline cancels this stream."""

        run = await self.store.load_run(self.claim.run_id)
        if (
            run is None
            or run.status is not EvalRunStatus.RUNNING
            or run.ownership is None
            or run.ownership.epoch != self.claim.epoch
            or run.scenario_progress is None
            or run.scenario_progress.attempt != self.claim.epoch
        ):
            raise EvalRunClaimLost("Scenario claim changed before trial settlement.")
        trial = run.scenario_progress.trials[self.trial_number - 1]
        if trial.phase in {
            EvalScenarioTrialPhase.COMPLETED,
            EvalScenarioTrialPhase.ERROR,
        }:
            return
        self.next_sequence = trial.next_event_sequence
        await self._update(
            EvalScenarioTrialPhase.ERROR,
            failure_code=EvalScenarioTrialFailureCode.EXECUTION_FAILED,
        )

    async def __call__(self, _: RunRequest) -> AsyncIterator[Event]:
        compiled = compile_eval_scenario(self.scenario)
        try:
            if self.initial_progress.phase in {
                EvalScenarioTrialPhase.AWAITING_APPROVAL,
                EvalScenarioTrialPhase.AWAITING_RESUME,
            }:
                if not 1 <= self.next_sequence <= len(self.scenario.events):
                    self._unexpected("Durable scenario checkpoint has an invalid event cursor.")
                step = self.scenario.events[self.next_sequence - 1]
                historical_events = await self.target.app.session_store.load_events(self.session_id)
                if not historical_events:
                    self._unexpected("Durable scenario session has no retained event history.")
                for historical_event in historical_events:
                    yield historical_event
                if self.initial_progress.phase is EvalScenarioTrialPhase.AWAITING_APPROVAL:
                    if type(step) is not ScenarioApprovalCheckpointEventV2:
                        self._unexpected("Durable scenario checkpoint is not an approval event.")
                    stream = await self._wait_for_approval(step, checkpoint_persisted=True)
                else:
                    if type(step) is not ScenarioResumedInputEventV2:
                        self._unexpected("Durable scenario checkpoint is not a resumed event.")
                    stream = (
                        await self._user_input_stream(step, checkpoint_persisted=True)
                        if step.resume_kind == "user_input"
                        else await self._resume_stream(step, checkpoint_persisted=True)
                    )
            else:
                initial_messages = _scenario_messages(
                    compiled.initial.input,
                    self.scenario,
                    self.binding,
                )
                request = self.target.request_base.model_copy(
                    update={
                        "session_id": self.session_id,
                        "messages": [
                            *(
                                ()
                                if self.external_envelope is None
                                else (self.external_envelope.message(),)
                            ),
                            *self.target.bootstrap_messages,
                            *initial_messages,
                        ],
                    },
                    deep=True,
                )
                await self._enter_running_phase()
                self.next_sequence = 1
                stream = self.target.app._run_private(
                    request,
                    expected_execution_profile=self.expected_execution_profile,
                )
            while True:
                async for emitted in self._drain(stream):
                    yield emitted
                session = await self.target.app.session_store.load(self.session_id)
                if session is None:
                    self._unexpected("Scenario session disappeared after runtime execution.")
                if self.next_sequence >= len(self.scenario.events):
                    if session.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                        self._unexpected("Scenario ended with a non-terminal root session.")
                    await self._update(EvalScenarioTrialPhase.COMPLETED)
                    return
                step = self.scenario.events[self.next_sequence]
                if type(step) is ScenarioApprovalCheckpointEventV2:
                    stream = await self._wait_for_approval(step)
                elif type(step) is ScenarioResumedInputEventV2:
                    stream = (
                        await self._user_input_stream(step)
                        if step.resume_kind == "user_input"
                        else await self._resume_stream(step)
                    )
                elif type(step) is ScenarioQueuedInputEventV2:
                    self._unexpected("Queued scenario input was not accepted by an active run.")
                else:
                    self._unexpected("Scenario contains an unexpected repeated initial input.")
        except asyncio.CancelledError:
            raise
        except (EvalRunClaimLost, EvalRunStateConflict):
            raise
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, ScenarioExecutionError)
                else (
                    EvalScenarioTrialFailureCode.EXECUTION_PROFILE_CHANGED
                    if isinstance(
                        exc,
                        EvalExecutionProfileChangedError | ExecutionProfileMismatchError,
                    )
                    else EvalScenarioTrialFailureCode.EXECUTION_FAILED
                )
            )
            with contextlib.suppress(EvalRunClaimLost, EvalRunStateConflict):
                await self._update(EvalScenarioTrialPhase.ERROR, failure_code=code)
            raise


async def run_compiled_eval_scenario(
    target: CorpusTarget,
    compiled: CompiledCorpusSuite,
    scenario: EvalScenarioDocumentV2,
    binding: ScenarioLaunchBindingV2,
    *,
    store: EvalStore,
    claim: EvalRunClaim,
    max_concurrency: int,
    poll_seconds: float,
    manifest_project_root: Path | None = None,
    expected_app_manifest_fingerprint: str | None = None,
    expected_execution_profile: ExecutionProfileIdentity | None = None,
    execution_capacity: EvalExecutionCapacity | None = None,
) -> CorpusExecutionResult:
    """Execute all scenario trials and return the ordinary corpus result shape."""

    if len(compiled.suite.cases) != 1 or compiled.trials != binding.trials:
        raise ValueError("Controlled scenario execution requires its derived one-case corpus.")
    if (
        expected_execution_profile is not None
        and type(expected_execution_profile) is not ExecutionProfileIdentity
    ):
        raise TypeError(
            "expected_execution_profile must be an exact ExecutionProfileIdentity or None."
        )
    if execution_capacity is not None and type(execution_capacity) is not EvalExecutionCapacity:
        raise TypeError("execution_capacity must be an exact EvalExecutionCapacity or None.")
    target_before = evaluation_target_identity(target, project_root=manifest_project_root)
    if (
        expected_app_manifest_fingerprint is not None
        and target_before.app_manifest_fingerprint != expected_app_manifest_fingerprint
    ):
        raise EvalExecutionProfileChangedError(
            "Scenario target manifest does not match its registered identity."
        )
    current = await store.load_run(claim.run_id)
    if (
        current is None
        or current.status is not EvalRunStatus.RUNNING
        or current.ownership is None
        or current.ownership.epoch != claim.epoch
    ):
        raise EvalRunClaimLost("Scenario claim is no longer current.")
    progress = current.scenario_progress
    if progress is None:
        initialized = await store.initialize_scenario_progress(
            claim,
            EvalScenarioRunProgress.create(
                scenario_revision=scenario.revision,
                binding_revision=binding.revision,
                attempt=claim.epoch,
                trials=tuple(
                    EvalScenarioTrialProgress(
                        trial_number=trial_number,
                        phase=EvalScenarioTrialPhase.PENDING,
                        next_event_sequence=0,
                    )
                    for trial_number in range(1, binding.trials + 1)
                ),
            ),
        )
        progress = initialized.scenario_progress
    if progress is None or progress.attempt != claim.epoch:
        raise EvalRunStateConflict("Scenario progress is unavailable for the current claim.")
    case = compiled.suite.cases[0]
    output_preview_bytes = min(
        EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
        PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES // binding.trials,
    )
    memory_attribution_bounds = eval_memory_attribution_bounds_for_trial_count(binding.trials)
    memory_attribution_source_limit = eval_memory_attribution_source_limit_for_trial_count(
        binding.trials
    )
    memory_attribution_max_bytes = eval_memory_attribution_max_bytes_for_trial_count(binding.trials)
    slots = [None] * binding.trials
    semaphore = asyncio.Semaphore(max_concurrency)
    memory_attribution_read_lifecycle = _FreshMemoryAttributionReadLifecycle(
        max_operations=max_concurrency
    )
    external_trials: tuple[ExternalTrialIdentityV1, ...] = ()
    if target.external_process is not None:
        external_process = target.external_process
        case_revision = compiled.corpus.cases[0].revision
        external_trials = tuple(
            ExternalTrialIdentityV1.create(
                native_run_id=claim.run_id,
                target_key=target.key,
                target_revision=external_process.revision,
                corpus_revision=compiled.corpus.revision,
                suite_id=compiled.suite.id,
                suite_revision=compiled.run_contract.suite_revision,
                case_id=case.id,
                case_revision=case_revision,
                trial_number=trial_number,
            )
            for trial_number in range(1, binding.trials + 1)
        )

    async def execute_trial(trial_number: int) -> None:
        driver = _ScenarioTrialDriver(
            target=target,
            scenario=scenario,
            binding=binding,
            store=store,
            claim=claim,
            trial_number=trial_number,
            initial_progress=progress.trials[trial_number - 1],
            poll_seconds=poll_seconds,
            manifest_project_root=manifest_project_root,
            expected_app_manifest_fingerprint=expected_app_manifest_fingerprint,
            expected_execution_profile=expected_execution_profile,
            external_envelope=(
                None
                if not external_trials
                else ExternalTrialEnvelopeV1(trial=external_trials[trial_number - 1])
            ),
        )
        async with semaphore:
            capacity_slot = (
                contextlib.nullcontext()
                if execution_capacity is None
                else execution_capacity.slot()
            )
            async with capacity_slot:
                execution = await _run_case_once_with_public_projection(
                    target.app,
                    case,
                    trial_number=trial_number,
                    suite_id=compiled.suite.id,
                    retain_final_output=False,
                    timeout_seconds=binding.timeout_seconds,
                    public_output_preview_bytes=output_preview_bytes,
                    memory_attribution_bounds=memory_attribution_bounds,
                    memory_attribution_source_limit=memory_attribution_source_limit,
                    memory_attribution_max_bytes=memory_attribution_max_bytes,
                    run_stream=driver,
                    memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
                )
                await driver.reconcile_terminal_progress()
                slots[trial_number - 1] = execution

    started_at = datetime.now(UTC)
    async with memory_attribution_read_lifecycle, asyncio.TaskGroup() as group:
        for trial_number in range(1, binding.trials + 1):
            group.create_task(execute_trial(trial_number))
    completed_at = datetime.now(UTC)
    executions = [item for item in slots if item is not None]
    if len(executions) != binding.trials:
        raise RuntimeError("Scenario execution lost a trial result.")
    trial_results = [item[0] for item in executions]
    public_data = tuple(item[1] for item in executions)
    if any(item is None for item in public_data):
        raise RuntimeError("Scenario execution lost a public trial projection.")
    case_result = _aggregate_trials(
        case,
        trial_results,
        started_at=started_at,
        completed_at=completed_at,
    )
    internal_run = EvalRun(
        suite_id=compiled.suite.id,
        status=aggregate_eval_status((case_result.status,)),
        score=aggregate_eval_score((case_result.score,)),
        cases=(case_result,),
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(int((completed_at - started_at).total_seconds() * 1000), 0),
        metadata=compiled.suite.metadata,
    )
    return await asyncio.to_thread(
        _finalize_compiled_corpus_result,
        target,
        compiled,
        target_before,
        internal_run,
        {case.id: public_data},
        manifest_project_root,
        external_trials,
    )


__all__ = [
    "ScenarioExecutionError",
    "corpus_for_eval_scenario",
    "run_compiled_eval_scenario",
    "scenario_launch_settings_from_invocation",
]
