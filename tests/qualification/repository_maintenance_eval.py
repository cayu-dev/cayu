"""Opt-in coding-stage Evals; the context owns every constructed deployment."""

import asyncio
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

from domain.coding_product import CodingProductTask  # ty: ignore[unresolved-import]
from domain.maintenance_case import SEED_BASE_REVISION  # ty: ignore[unresolved-import]
from evals.maintenance_seed import materialize_seed_repository  # ty: ignore[unresolved-import]
from operations.maintenance_requests import (  # ty: ignore[unresolved-import]
    capture_accepted_request,
)
from workflows.maintenance_coding import (  # ty: ignore[unresolved-import]
    MaintenanceCodingWorkflow,
)

from cayu import (
    CausalBudgetCostSummary,
    CorpusExecutionLimits,
    CorpusExecutionResult,
    EvalPlan,
    EvalSuiteTrialPolicyV1,
    Message,
    RunRequest,
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalInvocation,
    WorkflowEvalResult,
    WorkflowEvalTarget,
    copy_price_book,
    current_execution_deadline,
    run_eval_plan,
)
from cayu.evals import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
    eval_run_contract_for_corpus,
    pricing_profile_identity,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    FinalOutputEqualsAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
)
from tests.qualification.repository_maintenance_deployment import build_maintenance_deployment
from tests.qualification.repository_maintenance_identity import MaintenanceRunIntent
from tests.qualification.repository_maintenance_lifetime import (
    close_deployment,
    raise_lifetime_failures,
    wait_owned_task,
)
from tests.qualification.repository_maintenance_request import (
    bounded_text,
    copy_request,
    decode_request,
    encode_request,
)
from tests.qualification.repository_maintenance_results import coding_task_from_identity


def _configured_deployment(*, workspace_root):
    from configuration.maintenance import (  # ty: ignore[unresolved-import]
        configured_maintenance_budget,
    )

    return build_maintenance_deployment(
        budget_policy=configured_maintenance_budget(),
        workspace_root=workspace_root,
    )


@dataclass(frozen=True, repr=False)
class MaintenanceEvaluationSource:
    """Private retained source and its owned preparation, never a reset capability."""

    root: Path
    _preparation: asyncio.Task[str]

    async def wait_prepared(self):
        if await wait_owned_task(self._preparation) != SEED_BASE_REVISION:
            raise ValueError("Maintenance evaluation seed has an unexpected base revision.")

    @property
    def prepared(self):
        task = self._preparation
        return (
            task.done()
            and not task.cancelled()
            and task.exception() is None
            and task.result() == SEED_BASE_REVISION
        )


async def _prepare_source(sources, directory):
    root = Path(tempfile.mkdtemp(prefix="cayu-maintenance-eval-", dir=directory)) / "repository"
    source = MaintenanceEvaluationSource(
        root,
        asyncio.create_task(asyncio.to_thread(materialize_seed_repository, root)),
    )
    sources.append(source)
    await source.wait_prepared()
    return source


@dataclass(frozen=True, repr=False)
class MaintenanceEvaluationAttempt:
    """Private native trial identifiers; not a receipt or permission to recover."""

    run_id: str
    suite_id: str
    case_id: str
    trial_number: int
    workflow_run_id: str
    idempotency_key: str

    @classmethod
    def observe(cls, invocation):
        if type(invocation) is not WorkflowEvalInvocation:
            raise ValueError("Invalid maintenance evaluation invocation.")
        number = invocation.trial_number
        if type(number) is not int or not 1 <= number <= 100:
            raise ValueError("Invalid maintenance evaluation trial number.")
        return cls(
            **{
                field: bounded_text(getattr(invocation, field), bound=1024)
                for field in ("run_id", "suite_id", "case_id", "workflow_run_id", "idempotency_key")
            },
            trial_number=number,
        )


def _project_coding_result(evidence):
    payload = evidence.completion_event.payload
    return WorkflowEvalResult(
        final_output=payload["verdict"],
        structured_output={
            "product_run_id": payload["product_run_id"],
            "result_digest": payload["result_digest"],
            "verdict": payload["verdict"],
        },
    )


def maintenance_coding_corpus(*, source, reviewed_instruction):
    """Freeze the reviewed input and source once, then reuse it for both versions.

    This does not review, redact or authenticate private captured material.
    The caller must perform that review before supplying portable source/input.
    """
    suite = EvalSuiteSpec.create(
        id="maintenance-coding",
        name="Maintenance coding-stage regression",
        trial_request=TrialRequestSpec(
            trials=2,
            timeout_seconds=180,
            trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=2, max_concurrency=1),
        ),
    )
    case = EvalCaseSpec.create(
        id="inclusive-endpoint",
        suite_id=suite.id,
        name="Independent inclusive-endpoint acceptance",
        source=source,
        input=RunInputSpec(
            messages=(
                CorpusUserMessageSpec(
                    text=bounded_text(reviewed_instruction, bound=4096),
                ),
            )
        ),
        assertions=(FinalOutputEqualsAssertionSpec(id="independent-verdict", expected="verified"),),
    )
    return EvalCorpusDocument.create(
        target_key="maintenance-coding",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )


class MaintenanceEvaluationScope:
    """Retain native deployment owners, including after failed context exit.

    Keep this handle until all deployments have positively settled. Its tuple
    projection does not grant retry, redispatch or recovery authority.
    """

    def __init__(self, *, source_directory, **options):
        directory = Path(source_directory).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Maintenance evaluation source directory must exist.")
        self._owned = []
        self._sources = []
        self._attempts = []
        self._manager = _maintenance_eval_plan(
            owned=self._owned,
            sources=self._sources,
            attempts=self._attempts,
            source_directory=directory,
            **options,
        )

    @property
    def deployments(self):
        return tuple(self._owned)

    @property
    def sources(self):
        return tuple(self._sources)

    @property
    def attempts(self):
        return tuple(self._attempts)

    async def __aenter__(self):
        return await self._manager.__aenter__()

    async def __aexit__(self, *failure):
        return await self._manager.__aexit__(*failure)


def maintenance_eval_plan(**options):
    """Construct a retained scope; enter it to obtain the native corpus EvalPlan."""
    return MaintenanceEvaluationScope(**options)


@dataclass(frozen=True, repr=False)
class MaintenanceCorpusObservation:
    """Private same-execution host evidence, not authentication or recovery authority."""

    result_json: str
    attempts: tuple[MaintenanceEvaluationAttempt, ...]

    @property
    def result(self):
        return corpus_execution_result_from_json(self.result_json)

    async def inspect_costs(self, app, corpus, pricing):
        """Read recorded cohort estimates in one store/alias namespace.

        Use only the original trusted host observation, before its scope closes.
        This does not authenticate uploaded data or establish billed completeness.
        """
        try:
            result = self.result
            contract = eval_run_contract_for_corpus(corpus, result.run.suite_id)
            pricing = copy_price_book(pricing)
            pricing_id = pricing_profile_identity(pricing).fingerprint
            if contract.pricing_profile_fingerprint not in (None, pricing_id):
                raise ValueError
            for field in (
                "corpus_revision",
                "target_key",
                "suite_id",
                "suite_revision",
                "evidence_policy_revision",
                "pricing_profile_fingerprint",
                "trial_policy",
            ):
                if getattr(contract, field) != getattr(result.run, field):
                    raise ValueError
            if tuple((case.case_id, case.case_revision) for case in result.run.cases) != tuple(
                (case.case_id, case.case_revision) for case in contract.cases
            ):
                raise ValueError
            roster = {
                (case.case_id, number)
                for case in contract.cases
                for number in range(1, contract.trials + 1)
            }
            if any(len(case.trials) != contract.trials for case in result.run.cases):
                raise ValueError
            bindings = {}
            run_ids = set()
            if type(self.attempts) is not tuple or len(self.attempts) > 10000:
                raise ValueError
            for attempt in self.attempts:
                if type(attempt) is not MaintenanceEvaluationAttempt:
                    raise ValueError
                values = {
                    field: bounded_text(getattr(attempt, field), bound=1024)
                    for field in (
                        "run_id",
                        "suite_id",
                        "case_id",
                        "workflow_run_id",
                        "idempotency_key",
                    )
                }
                if type(attempt.trial_number) is not int:
                    raise ValueError
                key = (values["case_id"], attempt.trial_number)
                binding = (values["workflow_run_id"], values["idempotency_key"])
                if key not in roster or values["suite_id"] != contract.suite_id:
                    raise ValueError
                if key in bindings and bindings[key] != binding:
                    raise ValueError
                bindings[key] = binding
                run_ids.add(values["run_id"])
            if len(run_ids) > 1 or len({value[0] for value in bindings.values()}) != len(bindings):
                raise ValueError
        except (TypeError, ValueError, AttributeError):
            raise ValueError("Invalid maintenance cohort cost inputs.") from None

        sessions = {}
        missing = empty = observed = 0
        for root, _key in bindings.values():
            causal_id = app.project_causal_budget_id_for_exposure(root, session_ids=(root,))
            try:
                summary = await app.get_causal_budget_cost(causal_id, pricing, currency="USD")
            except KeyError:
                missing += 1
                continue
            validated = _validated_cohort_cost(summary, causal_id)
            observed += 1
            empty += not any(item.line_items for item in validated.session_costs)
            for item in validated.session_costs:
                if item.session_id in sessions and sessions[item.session_id] != item:
                    raise ValueError("Conflicting maintenance cohort cost observations.")
                sessions[item.session_id] = item
        with localcontext() as context:
            context.prec = 512
            total = sum((item.total_cost for item in sessions.values()), Decimal(0))
        items = tuple(line for session in sessions.values() for line in session.line_items)
        return {
            "basis": "recorded_runtime_events",
            "billing_completeness": "not_established",
            "result_revision": result.revision,
            "corpus_revision": contract.corpus_revision,
            "pricing_fingerprint": pricing_id,
            "currency": "USD",
            "expected_trials": len(roster),
            "observed_trials": len(bindings),
            "unobserved_trials": len(roster) - len(bindings),
            "missing_roots": missing,
            "empty_roots": empty,
            "session_count": len(sessions),
            "known_estimated_total": str(total) if observed else None,
            "unpriced_line_items": sum(not item.priced for item in items),
            "unknown_hosted_calls": sum(item.web_search_outcome_unknown for item in items),
        }


def _validated_cohort_cost(summary, expected_causal_id):
    try:
        if type(summary) is not CausalBudgetCostSummary:
            raise ValueError
        summary = CausalBudgetCostSummary.model_validate(
            {field: getattr(summary, field) for field in CausalBudgetCostSummary.model_fields}
        )
        if (
            summary.causal_budget_id != expected_causal_id
            or summary.currency != "USD"
            or len(set(summary.session_ids)) != summary.session_count
            or len(summary.session_ids) != summary.session_count
            or tuple(item.session_id for item in summary.session_costs)
            != tuple(summary.session_ids)
        ):
            raise ValueError
        amounts = [summary.total_cost]
        for session in summary.session_costs:
            if session.currency != "USD" or any(
                item.currency != "USD" for item in session.line_items
            ):
                raise ValueError
            if session.priced_model_steps != sum(
                item.priced and item.model_step > 0 for item in session.line_items
            ):
                raise ValueError
            amounts.extend([session.total_cost, *(item.total_cost for item in session.line_items)])
        if any(
            not amount.is_finite()
            or amount < 0
            or len(amount.as_tuple().digits) > 128
            or abs(amount.as_tuple().exponent) > 128
            for amount in amounts
        ):
            raise ValueError
        with localcontext() as context:
            context.prec = 512
            if summary.total_cost != sum(
                (item.total_cost for item in summary.session_costs), Decimal(0)
            ):
                raise ValueError
            for session in summary.session_costs:
                if session.total_cost != sum(
                    (item.total_cost for item in session.line_items), Decimal(0)
                ):
                    raise ValueError
        for field in (
            "model_steps",
            "priced_model_steps",
            "unpriced_model_steps",
            "missing_usage_model_steps",
            "missing_pricing_model_steps",
            "unsupported_pricing_model_steps",
        ):
            if getattr(summary, field) != sum(
                getattr(item, field) for item in summary.session_costs
            ):
                raise ValueError
        return summary
    except (TypeError, ValueError, AttributeError):
        raise ValueError("Invalid maintenance cohort cost evidence.") from None


@asynccontextmanager
async def maintenance_corpus_execution(scope, corpus):
    """Run the native fixed suite and retain its result/attempt association.

    Keep the supplied scope after any failure for existing cleanup inspection.
    Consume/save private evidence inside this context, while stores remain open.
    This is not an entrance for certifying caller-supplied reports or trial IDs.
    """
    if type(scope) is not MaintenanceEvaluationScope:
        raise ValueError("Maintenance corpus execution requires its evaluation scope.")
    async with scope as plan:
        result = await run_eval_plan(plan, corpus=corpus, suite_id="maintenance-coding")
        if type(result) is not CorpusExecutionResult:
            raise ValueError("Maintenance corpus execution returned an invalid result.")
        observation = MaintenanceCorpusObservation(
            result_json=corpus_execution_result_to_json(result),
            attempts=tuple(replace(attempt) for attempt in scope.attempts),
        )
        yield observation


@asynccontextmanager
async def _maintenance_eval_plan(
    *,
    owned,
    sources,
    attempts,
    source_directory,
    task,
    application_release_id,
    implementation_revision,
    result_projector_revision,
    execution_scope_revision,
    build_deployment=_configured_deployment,
):
    """Yield a native plan while retaining reference and per-trial resources.

    Execute and save private evidence inside this context. Supply actual reviewed
    release/revision pins; hashes do not authenticate untrusted reports. The
    trusted construction seam must use its supplied workspace_root and return
    a fresh deployment on every call with
    the same configuration and shared durable budget. It must not start work.
    Normal production construction uses the configured PostgreSQL deployment.

    Use maintenance_coding_corpus and native run_eval_plan: its two sequential
    trials and 180-second ceiling are enforced by the corpus compiler.
    This target performs coding, never Git/PR delivery or approval. A passed
    coding verdict is not a verified PR completion. The context is not a CLI
    no-argument factory: callers must retain it through native evidence reads.
    """
    if type(task) is not CodingProductTask:
        raise ValueError("Maintenance Evals requires an exact coding task.")
    template = replace(task)
    primary = None
    try:
        reference_source = await _prepare_source(sources, source_directory)
        reference = build_deployment(workspace_root=reference_source.root)
        owned.append(reference)
        await reference.validate_startup_schema()
        application = reference.application
        accepted = await capture_accepted_request(application, template)
        request = RunRequest(
            agent_name=application.agent_name,
            environment_name="coding",
            messages=[Message.text("user", accepted.instruction)],
        )
        messages = tuple(request.messages)

        async def factory(invocation):
            attempts.append(MaintenanceEvaluationAttempt.observe(invocation))
            if invocation.messages != messages:
                raise ValueError("Maintenance Evals input differs from the accepted task.")
            deadline = current_execution_deadline()
            if deadline is None or deadline.expires_at is None:
                raise ValueError("Maintenance Evals requires a finite coding deadline.")
            remaining = (deadline.expires_at - datetime.now(UTC)).total_seconds()
            if not 0 < remaining <= 180:
                raise ValueError("Maintenance Evals requires a coding deadline within 180 seconds.")
            source = await _prepare_source(sources, source_directory)
            deployment = build_deployment(workspace_root=source.root)
            owned.append(deployment)
            await deployment.validate_startup_schema()
            candidate = deployment.application
            expected = copy_request(
                accepted.model_copy(update={"repository_root": str(source.root)})
            )
            if await capture_accepted_request(candidate, template) != expected:
                raise ValueError("Maintenance Evals configuration changed between trials.")
            identity = await deployment.reservations.reserve(
                MaintenanceRunIntent(
                    tenant="maintenance-evaluation",
                    subject="maintenance-evaluator",
                    idempotency_key=invocation.idempotency_key,
                    request_json=encode_request(expected),
                ),
                workflow_session_id=invocation.workflow_run_id,
                coding_expires_at=deadline.expires_at.isoformat(),
            )

            async def quiesce():
                if await deployment.quiesce(timeout_s=30) is not True:
                    raise RuntimeError("Maintenance evaluation work has not settled.")

            return WorkflowEvalExecution(
                app=candidate.app,
                workflow=MaintenanceCodingWorkflow(
                    candidate,
                    coding_task_from_identity(identity),
                    accepted=decode_request(identity.intent.request_json),
                ),
                close=quiesce,
            )

        target = WorkflowEvalTarget(
            key="maintenance-coding",
            app=application.app,
            request_base=request.model_copy(update={"messages": []}),
            application_release_id=application_release_id,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            limits=CorpusExecutionLimits(
                max_cases=1,
                max_trials=2,
                max_timeout_seconds=180,
                max_concurrency=1,
            ),
            workflow_spec=MaintenanceCodingWorkflow.spec,
            implementation_revision=implementation_revision,
            result_projector_revision=result_projector_revision,
            execution_scope_revision=execution_scope_revision,
            instance_scope=WorkflowEvalInstanceScope.PER_TRIAL,
            workflow_factory=factory,
            result_projector=_project_coding_result,
            application_context={
                "stage": "coding",
                "corpus_revision": accepted.corpus_fingerprint,
                "probe_revision": accepted.probe_fingerprint,
            },
        )
        yield EvalPlan(workflow_target=target)
    except BaseException as exc:
        primary = exc

    async def close_all():
        failures = []
        for deployment in reversed(owned):
            try:
                await close_deployment(deployment)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("Maintenance evaluation cleanup failed.", failures)

    try:
        # Keep observer cancellation outside the multi-owner failure aggregate.
        # Every deployment still settles before the original signal propagates.
        await wait_owned_task(asyncio.create_task(close_all()))
    except BaseException as cleanup:
        if primary is not None:
            raise_lifetime_failures(primary, cleanup)
        raise
    if primary is not None:
        raise primary
