"""A corpus target's eval reviewer approves an agent's answers inside eval trials only."""

from __future__ import annotations

import asyncio
import html
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.evals.test_corpus_execution import _model_judge_target

import cayu.evals.runner as runner_module
from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolContext, ToolResult, ToolSpec
from cayu.approvals import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy
from cayu.evals import (
    EVAL_REVIEWER_MAX_APPROVALS,
    ApplicationEvaluator,
    ApproveAsProposedReviewer,
    EvalCase,
    EvalSuite,
    EventOccurred,
    FinalOutputContains,
    LLMJudge,
    ScriptedModelProvider,
    SessionStatusIs,
    ToolNotCalled,
    run_eval_suite,
)
from cayu.evals.benchmark_inspection import benchmark_failure_category
from cayu.evals.benchmark_rescore import _saved_evidence
from cayu.evals.corpus import (
    ApplicationEvaluatorAssertionSpec,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputContainsAssertionSpec,
    ModelJudgeAssertionSpec,
    ProcessEventAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    ToolCalledAssertionSpec,
    TrialRequestSpec,
)
from cayu.evals.eval_reviewer import EvalReviewerScopeError, _TrialReviewer
from cayu.evals.execution import (
    CorpusTarget,
    WorkflowEvalTarget,
    _run_compiled_corpus_trials,
    compile_corpus_suite,
    evaluation_target_identity,
    run_corpus_suite,
)
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    corpus_execution_compatibility,
)
from cayu.evals.execution_profiles import _eval_execution_target_material_identity
from cayu.evals.execution_reporting import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
    render_corpus_execution_html,
)
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime import RequireFinalTool
from cayu.sessions import InMemorySessionStore, SessionStore
from cayu.sessions._run_identity import eval_trial_scope
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.storage import SQLiteSessionStore
from cayu.tools import AlwaysRequireApprovalToolPolicy
from cayu.workflows.base import WorkflowSpec

SUBMIT_ANSWER = "submit_answer"
REVIEWER = "answer-reviewer"
REVIEW_PURPOSE = "answer-review"
AGENT = "analyst"
TARGET_KEY = "answer-analyst"
SUITE_ID = "questions"
ANSWER = "42"


class SubmitAnswerTool(Tool):
    """The analyst's final answer, which runs only once someone approved it."""

    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name=SUBMIT_ANSWER,
                description="Submit the final answer for review.",
                input_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                max_terminal_payload_bytes=1024,
            )
        )
        self.answers: list[str] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.answers.append(args["answer"])
        return ToolResult(content=f"Answer approved: {args['answer']}")


class AnswerReview(HumanReviewPolicy):
    """Shows the answer reviewer each submitted answer, for the answer-review purpose."""

    version = "answer-review.1"
    binding_key = b"test-fixture-only-binding-key-32-bytes"

    def authorize(self, context, *, session_id, session_metadata, action):
        return context.purpose == REVIEW_PURPOSE and context.recipient == REVIEWER

    def project(self, context, source):
        return HumanReviewDisclosure(
            status="permitted",
            fields=tuple(
                HumanReviewField(
                    label="Answer", text=source.arguments_by_call[call.tool_call_id]["answer"]
                )
                for call in source.calls
            ),
            sensitive_content="application_attested",
        )


def submission(answer: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(name=SUBMIT_ANSWER, arguments={"answer": answer}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def final_text(answer: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(f"The answer is {answer}."),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def reviewed_steps(*answers: str) -> list[list[ModelStreamEvent]]:
    """One trial per answer: the analyst submits it and, once approved, states it."""

    return [step for answer in answers for step in (submission(answer), final_text(answer))]


def reviewer(*, max_approvals: int = 3, **options: Any) -> ApproveAsProposedReviewer:
    return ApproveAsProposedReviewer(subject=REVIEWER, max_approvals=max_approvals, **options)


def answer_target(
    steps: list[list[ModelStreamEvent]],
    *,
    eval_reviewer: ApproveAsProposedReviewer | None = None,
    human_review: bool = False,
    key: str = TARGET_KEY,
    loop_policies: tuple = (),
    session_store: SessionStore | None = None,
) -> tuple[CorpusTarget, SubmitAnswerTool]:
    """An analyst whose every answer waits for approval, as an eval target."""

    app = CayuApp(
        session_store=InMemorySessionStore() if session_store is None else session_store,
        enable_logging=False,
        human_review_policy=AnswerReview() if human_review else None,
    )
    app.register_provider(ScriptedModelProvider(steps), default=True)
    tool = SubmitAnswerTool()
    app.register_agent(
        AgentSpec(name=AGENT, model="fixture-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools={SUBMIT_ANSWER}),
    )
    target = CorpusTarget(
        key=key,
        app=app,
        request_base=RunRequest(
            agent_name=AGENT, messages=[], max_steps=8, loop_policies=loop_policies
        ),
        application_release_id="analyst-release",
        eval_reviewer=eval_reviewer,
    )
    return target, tool


def answer_corpus(
    *, trials: int = 2, target_key: str = TARGET_KEY, extra_assertions: tuple = ()
) -> EvalCorpusDocument:
    """One question whose answer must be 42, approved before it counts."""

    suite = EvalSuiteSpec.create(
        id=SUITE_ID,
        name="Questions",
        trial_request=TrialRequestSpec(trials=trials, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="meaning",
        suite_id=suite.id,
        name="Meaning",
        source=EvaluationSourceIdentityV1(
            application_release_id="captured-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "e" * 64,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="What is the answer?"),)),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputContainsAssertionSpec(id="answer", expected=ANSWER),
            ProcessEventAssertionSpec(id="requested", event="tool_approval_requested"),
            ProcessEventAssertionSpec(id="approved", event="tool_approved"),
            *extra_assertions,
        ),
    )
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )


async def approval_events(app: CayuApp) -> list[tuple[str, list[str], list[Any]]]:
    """Each session's approval requests and the resolvers of its approvals."""

    listed = await app.session_store.list_sessions()
    sessions = []
    for session in listed.sessions:
        events = await app.session_store.load_events(session.id)
        sessions.append(
            (
                session.id,
                [
                    event.type
                    for event in events
                    if event.type
                    in {EventType.TOOL_CALL_APPROVAL_REQUESTED, EventType.TOOL_CALL_APPROVED}
                ],
                [
                    event.payload.get("resolved_by")
                    for event in events
                    if event.type is EventType.TOOL_CALL_APPROVED
                ],
            )
        )
    return sessions


def outcomes(trial) -> dict[str, str]:
    return {assertion.assertion_id: assertion.outcome for assertion in trial.assertions}


@pytest.mark.parametrize("human_review", (False, True), ids=("approval", "human-review"))
def test_reviewer_approves_each_answer_so_trials_are_scored_on_their_merits(human_review):
    target, tool = answer_target(
        reviewed_steps(ANSWER, "41"),
        eval_reviewer=reviewer(review_purpose=REVIEW_PURPOSE if human_review else None),
        human_review=human_review,
    )

    result = asyncio.run(run_corpus_suite(target, answer_corpus(), SUITE_ID))

    right, wrong = result.run.cases[0].trials
    assert (right.status, wrong.status) == ("passed", "failed")
    assert outcomes(right) == dict.fromkeys(
        ("completed", "answer", "requested", "approved"), "passed"
    )
    # The wrong answer was approved as proposed and fails only the answer check.
    assert outcomes(wrong) == {
        "completed": "passed",
        "answer": "failed",
        "requested": "passed",
        "approved": "passed",
    }
    assert right.execution_failure_category is None
    assert wrong.execution_failure_category is None
    assert tool.answers == [ANSWER, "41"]
    sessions = asyncio.run(approval_events(target.app))
    assert len(sessions) == 2
    for _, approval_types, resolvers in sessions:
        assert approval_types == [
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            EventType.TOOL_CALL_APPROVED,
        ]
        assert resolvers == [{"subject": REVIEWER, "tenant": None, "source": "system"}]


def test_reviewer_continues_a_trial_under_its_request_loop_policies():
    # Request loop policies are part of the session's execution profile and are not
    # checkpointed, so each approval's continuation has to pass them again.
    target, tool = answer_target(
        reviewed_steps(ANSWER),
        eval_reviewer=reviewer(),
        loop_policies=(RequireFinalTool((SUBMIT_ANSWER,)),),
    )

    result = asyncio.run(run_corpus_suite(target, answer_corpus(trials=1), SUITE_ID))

    (trial,) = result.run.cases[0].trials
    assert trial.status == "passed"
    assert tool.answers == [ANSWER]


def test_without_a_reviewer_each_trial_stops_awaiting_approval_unscored():
    target, tool = answer_target([submission(ANSWER), submission(ANSWER)])

    result = asyncio.run(run_corpus_suite(target, answer_corpus(), SUITE_ID))

    assert result.run.status == "unavailable"
    trials = result.run.cases[0].trials
    for trial in trials:
        assert trial.status == "unavailable"
        assert trial.execution_failure_category == "awaiting_approval"
        assert trial.execution_limit is None
        assert trial.evidence_complete is True
        assert trial.code is EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE
        assert set(outcomes(trial).values()) == {"unavailable"}
        assert benchmark_failure_category(trial) == "awaiting_approval"
        assert _saved_evidence(trial).final_output_state == "unavailable"
    assert tool.answers == []
    presented = present_eval_result(result).cases[0].trials[0]
    assert presented.dimensions.runtime == "failed"
    assert presented.dimensions.candidate == "not_scored"
    assert "Candidate run: stopped awaiting approval." in html.unescape(
        render_corpus_execution_html(result)
    )
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(result)) == result
    for _, approval_types, resolvers in asyncio.run(approval_events(target.app)):
        assert approval_types == [EventType.TOOL_CALL_APPROVAL_REQUESTED]
        assert resolvers == []


@pytest.mark.parametrize("durable", (False, True), ids=("memory", "sqlite"))
def test_reviewer_approves_at_most_its_bound_then_the_trial_stops_awaiting_approval(
    durable, tmp_path
):
    bounded = reviewer(max_approvals=2)
    # The stop's evidence spans the run and both continuations, so a durable store must
    # match what the trial observed across all three.
    target, tool = answer_target(
        [submission("40"), submission("41"), submission(ANSWER)],
        eval_reviewer=bounded,
        session_store=SQLiteSessionStore(tmp_path / "sessions.sqlite") if durable else None,
    )
    compiled = compile_corpus_suite(answer_corpus(trials=1), target, SUITE_ID)

    run, _ = asyncio.run(
        runner_module._run_eval_suite_with_public_projection(
            target.app,
            compiled.suite,
            max_concurrency=1,
            case_timeout_seconds=30,
            trials=1,
            trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1),
            output_preview_bytes=1024,
            eval_reviewer=bounded,
        )
    )

    (trial,) = run.cases[0].trials
    assert trial.status == "unavailable"
    assert trial.evidence_complete is True
    assert trial.execution_failure_category == "awaiting_approval"
    assert trial.unavailable_reason == (
        "Not evaluated (awaiting_approval): the candidate's run stopped on a tool call "
        "awaiting approval; the eval reviewer had approved its limit of 2 calls."
    )
    assert tool.answers == ["40", "41"]
    ((_, approval_types, resolvers),) = asyncio.run(approval_events(target.app))
    assert approval_types.count(EventType.TOOL_CALL_APPROVAL_REQUESTED) == 3
    assert approval_types.count(EventType.TOOL_CALL_APPROVED) == 2
    assert len(resolvers) == 2
    if durable:
        asyncio.run(target.app.session_store.close())


def test_reviewer_leaves_an_answer_its_review_policy_does_not_show_it_pending():
    target, tool = answer_target(
        [submission(ANSWER)],
        eval_reviewer=reviewer(review_purpose="another-purpose"),
        human_review=True,
    )
    compiled = compile_corpus_suite(answer_corpus(trials=1), target, SUITE_ID)

    run, _ = asyncio.run(
        runner_module._run_eval_suite_with_public_projection(
            target.app,
            compiled.suite,
            max_concurrency=1,
            case_timeout_seconds=30,
            trials=1,
            trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1),
            output_preview_bytes=1024,
            eval_reviewer=target.eval_reviewer,
        )
    )

    (trial,) = run.cases[0].trials
    assert trial.execution_failure_category == "awaiting_approval"
    assert trial.unavailable_reason is not None
    assert trial.unavailable_reason.endswith(
        "the application's human review policy did not let the eval reviewer inspect it."
    )
    assert tool.answers == []


def test_reviewer_policy_and_bound_are_target_identity_and_break_comparability():
    def run(eval_reviewer):
        target, _ = answer_target(reviewed_steps(ANSWER, ANSWER), eval_reviewer=eval_reviewer)
        return target, asyncio.run(run_corpus_suite(target, answer_corpus(), SUITE_ID))

    three_target, three = run(reviewer(max_approvals=3))
    _, three_again = run(reviewer(max_approvals=3))
    two_target, two = run(reviewer(max_approvals=2))
    other_subject = ApproveAsProposedReviewer(subject="another-reviewer", max_approvals=3)

    assert three.target.eval_reviewer == reviewer(max_approvals=3).identity()
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(three)) == three
    assert three.target == evaluation_target_identity(three_target)
    assert three.target.eval_reviewer.model_dump(mode="json") == {
        "schema_version": 1,
        "revision": three.target.eval_reviewer.revision,
        "policy": "approve_as_proposed",
        "subject": REVIEWER,
        "max_approvals": 3,
    }
    assert (
        len(
            {
                three.target.eval_reviewer.revision,
                two.target.eval_reviewer.revision,
                other_subject.identity().revision,
            }
        )
        == 3
    )
    assert corpus_execution_compatibility(three, three_again).comparable is True
    different = corpus_execution_compatibility(three, two)
    assert different.comparable is False
    assert different.reasons == (CorpusComparisonReason.EXTERNAL_TARGET_REVISION_MISMATCH,)
    # The server's execution profile commits the reviewer too.
    assert _eval_execution_target_material_identity(
        three_target
    ) != _eval_execution_target_material_identity(two_target)
    report = html.unescape(render_corpus_execution_html(three))
    assert (
        f"Eval reviewer: approve_as_proposed as {REVIEWER}, at most 3 approvals per trial" in report
    )

    unreviewed_target, _ = answer_target([submission(ANSWER), submission(ANSWER)])
    unreviewed = asyncio.run(run_corpus_suite(unreviewed_target, answer_corpus(), SUITE_ID))
    assert "eval_reviewer" not in unreviewed.target.model_dump(mode="json")
    assert "Eval reviewer" not in render_corpus_execution_html(unreviewed)
    assert corpus_execution_compatibility(three, unreviewed).reasons == (
        CorpusComparisonReason.EXTERNAL_TARGET_REVISION_MISMATCH,
    )


def test_only_a_plain_approval_wait_counts_as_stopped_awaiting_approval():
    target, _ = answer_target([submission(ANSWER)])
    app = target.app

    async def served_stop():
        async for _ in app.run(
            RunRequest(
                agent_name=AGENT,
                session_id="paused-session",
                messages=[Message.text("user", "What is the answer?")],
            )
        ):
            pass
        session = await app.session_store.load("paused-session")
        return session, await app.session_store.load_events("paused-session")

    session, events = asyncio.run(served_stop())
    assert runner_module._stopped_awaiting_approval(session, events) is True
    # A continuation that failed after an approval, or a recovery, records why it left
    # the session waiting again; that keeps its ordinary outcome.
    for failure in (
        {"error": "environment reconnect failed", "error_type": "RuntimeError"},
        {"failure_evidence": {"exception_types": ["RuntimeError"]}},
        {"recovered": True},
    ):
        terminal = events[-1].model_copy(update={"payload": {**events[-1].payload, **failure}})
        assert runner_module._stopped_awaiting_approval(session, [*events[:-1], terminal]) is False


def test_reviewer_cannot_approve_outside_its_eval_trial():
    target, tool = answer_target(
        [*reviewed_steps(ANSWER), submission(ANSWER)], eval_reviewer=reviewer()
    )
    app = target.app
    # The target's reviewer approves its own eval trials...
    result = asyncio.run(run_corpus_suite(target, answer_corpus(trials=1), SUITE_ID))
    assert result.run.status == "passed"

    async def served_run():
        # ...but a session the application serves still waits for a person.
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name=AGENT,
                    session_id="served-session",
                    messages=[Message.text("user", "What is the answer?")],
                )
            )
        ]
        assert events[-1].payload["interruption_type"] == "tool_approval_required"

        async def no_events():
            return
            yield

        assert target.eval_reviewer is not None
        trial_reviewer = _TrialReviewer(app, target.eval_reviewer, "served-session")
        with pytest.raises(EvalReviewerScopeError):
            async for _ in trial_reviewer.review(no_events()):
                pass
        # Inside an eval trial's scope it still refuses a session no trial created.
        async with eval_trial_scope():
            with pytest.raises(EvalReviewerScopeError):
                async for _ in trial_reviewer.review(no_events()):
                    pass
        session = await app.session_store.load("served-session")
        pending = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="served-session", kind=PendingActionKind.TOOL_APPROVAL)
        )
        events = await app.session_store.load_events("served-session")
        return session, pending, events

    session, pending, events = asyncio.run(served_run())
    assert session.status is SessionStatus.INTERRUPTED
    assert len(pending.actions) == 1
    assert not any(event.type is EventType.TOOL_CALL_APPROVED for event in events)
    assert tool.answers == [ANSWER]


def test_a_corpus_cannot_supply_or_switch_on_a_reviewer():
    document = answer_corpus().model_dump(mode="json")
    declared = reviewer().model_dump(mode="json")

    with pytest.raises(ValidationError, match="eval_reviewer"):
        EvalCorpusDocument.model_validate({**document, "eval_reviewer": declared})
    suite = {**document["suites"][0]}
    suite["trial_request"] = {**suite["trial_request"], "eval_reviewer": declared}
    with pytest.raises(ValidationError, match="eval_reviewer"):
        EvalCorpusDocument.model_validate({**document, "suites": [suite]})


def test_only_a_valid_reviewer_on_an_ordinary_corpus_target_is_accepted():
    for invalid in (
        {"subject": "cayu:approval-expiry", "max_approvals": 1},
        {"subject": REVIEWER, "max_approvals": 0},
        {"subject": REVIEWER, "max_approvals": EVAL_REVIEWER_MAX_APPROVALS + 1},
        {"subject": " ", "max_approvals": 1},
    ):
        with pytest.raises(ValidationError):
            ApproveAsProposedReviewer.model_validate(invalid)
    with pytest.raises(TypeError, match="eval_reviewer must be an exact"):
        answer_target([], eval_reviewer=cast("Any", {"subject": REVIEWER, "max_approvals": 1}))
    # A human review policy without a dashboard purpose needs the reviewer's purpose.
    with pytest.raises(ValidationError, match="needs review_purpose"):
        answer_target([], eval_reviewer=reviewer(), human_review=True)
    target, _ = answer_target([])
    with pytest.raises(ValidationError, match="cannot declare an eval reviewer"):
        WorkflowEvalTarget(
            key="answer-workflow",
            app=target.app,
            request_base=target.request_base,
            application_release_id="analyst-release",
            workflow_spec=WorkflowSpec(name="answer"),
            implementation_revision="sha256:" + "1" * 64,
            result_projector_revision="sha256:" + "2" * 64,
            execution_scope_revision="sha256:" + "3" * 64,
            workflow_factory=cast("Any", lambda invocation: None),
            result_projector=cast("Any", lambda evidence: None),
            eval_reviewer=reviewer(),
        )


def test_a_case_that_expects_the_approval_stop_is_scored_on_it():
    target, _ = answer_target([submission(ANSWER)] * 4)
    expects_stop = ProcessEventAssertionSpec(id="stopped", event="session_interrupted", min_count=1)
    corpus = answer_corpus(trials=1, extra_assertions=(expects_stop,))

    result = asyncio.run(run_corpus_suite(target, corpus, SUITE_ID))

    (trial,) = result.run.cases[0].trials
    assert trial.execution_failure_category == "awaiting_approval"
    assert trial.status == "failed"
    assert outcomes(trial) == {
        "completed": "failed",
        "answer": "failed",
        "requested": "passed",
        "approved": "failed",
        "stopped": "passed",
    }
    assert present_eval_result(result).cases[0].trials[0].dimensions.candidate == "failed"

    # A direct suite that expects the pause keeps scoring it, failures included.
    for expects_pause in (
        SessionStatusIs("interrupted"),
        EventOccurred(EventType.SESSION_INTERRUPTED),
    ):
        direct = asyncio.run(
            run_eval_suite(target.app, direct_suite(expects_pause, FinalOutputContains(ANSWER)))
        )
        (direct_trial,) = direct.cases[0].trials
        assert direct_trial.status == "failed"
        assert direct_trial.execution_failure_category == "awaiting_approval"
        assert [assertion.outcome for assertion in direct_trial.assertions] == ["passed", "failed"]


def direct_suite(*assertions) -> EvalSuite:
    return EvalSuite(
        id="pauses",
        cases=[
            EvalCase(
                id="pauses-for-review",
                request=RunRequest(
                    agent_name=AGENT, messages=[Message.text("user", "What is the answer?")]
                ),
                assertions=list(assertions),
            )
        ],
    )


def test_checks_that_hold_on_an_approval_stop_still_pass():
    target, tool = answer_target([submission(ANSWER)] * 3)
    app = target.app

    # "Asks for approval before the answer runs" holds on the stop, so it passes...
    gate = asyncio.run(
        run_eval_suite(
            app,
            direct_suite(
                EventOccurred(EventType.TOOL_CALL_APPROVAL_REQUESTED), ToolNotCalled(SUBMIT_ANSWER)
            ),
        )
    )
    (gate_trial,) = gate.cases[0].trials
    assert gate_trial.status == "passed"
    assert gate_trial.execution_failure_category == "awaiting_approval"

    # ...as it does in a corpus...
    suite = EvalSuiteSpec.create(
        id=SUITE_ID,
        name="Gate",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = (
        answer_corpus()
        .cases[0]
        .model_copy(
            update={
                "assertions": (
                    ProcessEventAssertionSpec(id="requested", event="tool_approval_requested"),
                    ToolCalledAssertionSpec(
                        id="not-run", tool_name=SUBMIT_ANSWER, max_count=0, min_count=0
                    ),
                )
            }
        )
    )
    corpus = EvalCorpusDocument.create(
        target_key=TARGET_KEY,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(
            EvalCaseSpec.create(
                id=case.id,
                suite_id=suite.id,
                name=case.name,
                source=case.source,
                input=case.input,
                assertions=case.assertions,
            ),
        ),
    )
    corpus_result = asyncio.run(run_corpus_suite(target, corpus, SUITE_ID))
    (corpus_trial,) = corpus_result.run.cases[0].trials
    assert corpus_trial.status == "passed"
    assert corpus_trial.execution_failure_category == "awaiting_approval"
    assert benchmark_failure_category(corpus_trial) == "passed"
    # A scored stop keeps its output for static rescoring.
    assert _saved_evidence(corpus_trial).final_output_state == "complete"
    assert present_eval_result(corpus_result).cases[0].trials[0].dimensions.candidate == "passed"

    # ...but a check the stop fails is not a failed candidate.
    answer = asyncio.run(
        run_eval_suite(
            app,
            direct_suite(
                EventOccurred(EventType.TOOL_CALL_APPROVAL_REQUESTED), FinalOutputContains(ANSWER)
            ),
        )
    )
    (answer_trial,) = answer.cases[0].trials
    assert answer_trial.status == "unavailable"
    assert answer_trial.execution_failure_category == "awaiting_approval"
    assert {assertion.outcome for assertion in answer_trial.assertions} == {"unavailable"}
    assert tool.answers == []


def test_an_unscored_approval_stop_dispatches_no_model_judge():
    judge, judge_provider = _model_judge_target()
    target, _ = answer_target([submission(ANSWER)])
    target = target.model_copy(update={"model_judges": (judge,)})
    quality = ModelJudgeAssertionSpec(
        id="quality",
        evaluator_key=judge.key,
        rubric="Score whether the answer is correct.",
        rubric_version="quality-v1",
        threshold=0.7,
    )
    corpus = answer_corpus(trials=1, extra_assertions=(quality,))

    result = asyncio.run(run_corpus_suite(target, corpus, SUITE_ID))

    (trial,) = result.run.cases[0].trials
    assert trial.status == "unavailable"
    assert trial.execution_failure_category == "awaiting_approval"
    assert set(outcomes(trial).values()) == {"unavailable"}
    # The judge was never asked to judge an answer the run did not reach.
    assert judge_provider.requests == []


def test_an_unscored_approval_stop_dispatches_no_judge_an_application_evaluator_builds():
    judge, judge_provider = _model_judge_target()
    evaluator = ApplicationEvaluator(
        key="answer-judge",
        implementation_revision="sha256:" + "4" * 64,
        factory=lambda params: LLMJudge(
            judge.app, agent_name=judge.agent_name, rubric="Score whether the answer is 42."
        ),
    )
    target, _ = answer_target([submission(ANSWER)])
    target = target.model_copy(update={"application_evaluators": (evaluator,)})
    judged = ApplicationEvaluatorAssertionSpec(id="judged", evaluator_key=evaluator.key)

    result = asyncio.run(
        run_corpus_suite(target, answer_corpus(trials=1, extra_assertions=(judged,)), SUITE_ID)
    )

    (trial,) = result.run.cases[0].trials
    assert trial.status == "unavailable"
    assert trial.execution_failure_category == "awaiting_approval"
    assert judge_provider.requests == []


def test_trials_a_worker_claims_are_reviewed_like_a_whole_run():
    target, tool = answer_target(reviewed_steps(ANSWER, ANSWER), eval_reviewer=reviewer())
    compiled = compile_corpus_suite(answer_corpus(), target, SUITE_ID)

    class Claimed:
        def __init__(self, claims, case_id, trial_number):
            self.claims, self.case_id, self.trial_number = claims, case_id, trial_number

        async def settle(self, execute):
            self.claims.settled[(self.case_id, self.trial_number)] = await execute()

    class Claims:
        def __init__(self):
            self.open = [("meaning", 1), ("meaning", 2)]
            self.settled = {}

        async def claim(self):
            return Claimed(self, *self.open.pop(0)) if self.open else None

        async def wait(self):
            return False

    claims = Claims()
    asyncio.run(
        _run_compiled_corpus_trials(target, compiled, trial_claims=claims, max_concurrency=1)
    )

    assert sorted(claims.settled) == [("meaning", 1), ("meaning", 2)]
    for trial, _ in claims.settled.values():
        assert trial.status == "passed"
        assert trial.execution_failure_category is None
    assert tool.answers == [ANSWER, ANSWER]
