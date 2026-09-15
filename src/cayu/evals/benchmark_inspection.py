"""Bounded, read-only campaign inspection over native results and observations."""

from __future__ import annotations

import hashlib
import html
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import Field, StrictInt, StrictStr, model_validator

from cayu.evals._inspection_documents import decode_document, read_regular_file
from cayu.evals.benchmark_campaign import BenchmarkCampaignV1, load_benchmark_campaign
from cayu.evals.corpus import _content_revision, _PortableModel
from cayu.evals.published import PublishedEvalTrialResult, PublishedUsageSummaryV1
from cayu.evals.result_contract import EvalTrialOutputPreviewV1
from cayu.evals.session_inspection import EvalSessionInspectionV1, inspect_eval_sessions
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore

BENCHMARK_INSPECTION_MAX_BYTES = 64 * 1024 * 1024
BENCHMARK_OBSERVATION_MAX_FILES = 64


class BenchmarkTrialInspectionV1(_PortableModel):
    run_id: StrictStr
    case_id: StrictStr
    trial_number: StrictInt = Field(ge=1, le=100)
    run_status: StrictStr
    result_state: Literal["published", "provisional", "unavailable"]
    status: Literal["pending", "passed", "failed", "error", "unavailable", "cancelled"]
    failure_category: StrictStr
    diagnostic_code: StrictStr | None = None
    score: float | None = None
    source_trial_revision: StrictStr | None = None
    output: EvalTrialOutputPreviewV1 | None = None
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    usage: PublishedUsageSummaryV1 | None = None
    usage_availability: Literal["known", "partial", "unavailable"] = "unavailable"
    cost_availability: Literal["known", "partial", "unavailable"] = "unavailable"
    cost_limitation: StrictStr | None = "no_retained_priced_total"
    costs: tuple[dict[str, Any], ...] = ()
    evidence_state: Literal["complete", "incomplete", "unavailable"] = "unavailable"
    execution_status: StrictStr | None = None
    capture_diagnostic: dict[str, Any] | None = None
    failure_capture: dict[str, Any] | None = None
    failure_evidence: dict[str, Any] | None = None
    assertions: tuple[dict[str, Any], ...] = ()
    session_id: StrictStr | None = None
    session_sqlite_path: StrictStr | None = None
    session_association: Literal["result_revision", "provisional", "unavailable"] = "unavailable"
    session_evidence: EvalSessionInspectionV1 | None = None
    prior_session_references: tuple[dict[str, Any], ...] = ()
    limitations: tuple[StrictStr, ...] = ()
    dashboard_path: StrictStr


def _campaign_status(rows, policy):
    counts = Counter(row.status for row in rows)
    if counts["pending"]:
        return "pending"
    if counts["error"] or counts["unavailable"] or counts["cancelled"]:
        return "incomplete"
    if policy is None:
        raise ValueError("Published campaign outcomes require the admitted trial policy.")
    passed = Counter(row.case_id for row in rows if row.status == "passed")
    return (
        "passed"
        if all(passed[row.case_id] >= policy.minimum_passed_trials for row in rows)
        else "failed"
    )


class BenchmarkCampaignInspectionV1(_PortableModel):
    schema_version: Literal[1] = 1
    campaign: BenchmarkCampaignV1
    trial_policy: EvalSuiteTrialPolicyV1 | None = None
    status: Literal["pending", "passed", "failed", "incomplete"]
    counts: dict[str, int]
    trials: tuple[BenchmarkTrialInspectionV1, ...]
    limitations: tuple[StrictStr, ...] = ()
    successors: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_slots(self):
        exposure = self.campaign.runs[0].spec.invocation.authored_suite_exposure
        assert exposure is not None
        trials = exposure.candidate_trials // len(self.campaign.selection.cases)
        expected = {
            (run.spec.id, case_id, number)
            for run in self.campaign.runs
            for case_id in run.case_ids
            for number in range(1, trials + 1)
        }
        actual = {(row.run_id, row.case_id, row.trial_number) for row in self.trials}
        if actual != expected or len(actual) != len(self.trials):
            raise ValueError("Inspection rows must cover each exact admitted trial once.")
        counts: Counter[str] = Counter(row.status for row in self.trials)
        counts["total"] = len(self.trials)
        if {key: value for key, value in self.counts.items() if value} != dict(counts):
            raise ValueError("Inspection counts do not match its trial rows.")
        if self.trial_policy is not None and (
            self.trial_policy.revision != exposure.trial_policy_revision
            or self.trial_policy.trial_count != trials
        ):
            raise ValueError("Inspection trial policy does not match its admission.")
        expected_status = _campaign_status(self.trials, self.trial_policy)
        if self.status != expected_status:
            raise ValueError("Inspection status does not match its trial rows.")
        return self


def _observations(directory: Path, campaign: BenchmarkCampaignV1):
    """Each observing process has its own file, so concurrent resume cannot erase peers."""

    root = directory / "observations"
    observations: dict[tuple[str, int], list[dict[str, Any]]] = {}
    limitations = set()
    if not root.exists():
        return observations, {"trial_observations_missing"}
    if root.is_symlink():
        raise ValueError("Campaign observation directories cannot be symbolic links.")
    total = 0
    selected = {case.id for case in campaign.selection.cases}
    for index, child in enumerate(root.iterdir()):
        if index >= BENCHMARK_OBSERVATION_MAX_FILES:
            limitations.add("trial_observation_file_limit")
            break
        if child.is_symlink() or not child.is_dir():
            limitations.add("trial_observation_invalid_entry")
            continue
        raw = read_regular_file(
            child / "progress-0.json", max_bytes=BENCHMARK_INSPECTION_MAX_BYTES - total
        )
        if raw is None:
            continue
        total += len(raw)
        document = decode_document(raw)
        if (
            document.get("launch_id") != campaign.id
            or document.get("fingerprint") != campaign.revision
        ):
            limitations.add("trial_observation_identity_mismatch")
            continue
        records = document.get("trials")
        if type(records) is not list or len(records) > 100_000:
            limitations.add("trial_observation_invalid_records")
            continue
        for record in records:
            if type(record) is not dict:
                limitations.add("trial_observation_invalid_records")
                continue
            case_id, trial_number = record.get("case_id"), record.get("trial_number")
            if (
                type(case_id) is not str
                or case_id not in selected
                or type(trial_number) is not int
                or not 1 <= trial_number <= 100
            ):
                limitations.add("trial_observation_unknown_slot")
                continue
            observations.setdefault((case_id, trial_number), []).append(record)
    return observations, limitations


def benchmark_failure_category(trial: PublishedEvalTrialResult) -> str:
    """Map existing typed result codes without matching arbitrary exception messages."""

    code = trial.code.value
    if code == "recovery_reexecution_blocked":
        return "recovery_blocked"
    if code == "case_timeout":
        return "timeout"
    if trial.execution_failure_category is not None or trial.execution_status == "failed":
        return trial.execution_failure_category or "execution_failure"
    if code == "passed":
        return "passed"
    if code == "assertion_failed":
        return "answer_mismatch"
    if code == "assertion_evaluation_failed":
        return "scoring_failure"
    if code in {
        "workflow_capture_failed",
        "terminal_evidence_failed",
        "evidence_preparation_failed",
    }:
        return "capture_failure"
    if code in {"external_target_cancelled", "interrupted_evidence_unavailable"}:
        return "cancelled"
    if "unavailable" in code or code in {
        "workflow_completion_missing",
        "external_target_incomplete",
        "external_target_unknown",
    }:
        return "evidence_unavailable"
    return "execution_failure"


def _assertion_summary(assertion) -> dict[str, Any]:
    # Reference values, rubrics, private truth, tool JSON and judge explanations
    # do not belong in an inspection export. Their identities remain available.
    summary = {
        "id": assertion.assertion_id,
        "revision": assertion.assertion_revision,
        "kind": assertion.detail.kind,
        "outcome": assertion.outcome,
        "score": assertion.score,
    }
    if assertion.detail.kind in {"model_judge", "structured_model_judge"}:
        detail = assertion.detail.model_dump(mode="json")
        summary["judge"] = {
            key: detail[key] for key in ("diagnostic", "usage", "cost") if key in detail
        }
    return summary


def _cost_observations(trial):
    costs = {}
    if trial is None:
        return {
            "costs": (),
            "cost_availability": "unavailable",
            "cost_limitation": "no_retained_priced_total",
        }
    for assertion in trial.assertions:
        detail = assertion.detail
        if detail.kind != "max_estimated_cost" or detail.estimated_cost is None:
            continue
        cost = {
            "currency": detail.currency,
            "estimated_cost": detail.estimated_cost,
            "priced_model_steps": detail.priced_model_steps,
            "unpriced_model_steps": detail.unpriced_model_steps,
        }
        if detail.currency in costs and costs[detail.currency] != cost:
            return {
                "costs": (),
                "cost_availability": "unavailable",
                "cost_limitation": "conflicting_retained_costs",
            }
        costs[detail.currency] = cost
    partial = any(item["unpriced_model_steps"] for item in costs.values())
    return {
        "costs": tuple(costs[key] for key in sorted(costs)),
        "cost_availability": "partial" if partial else "known" if costs else "unavailable",
        "cost_limitation": "unpriced_model_steps"
        if partial
        else None
        if costs
        else "no_retained_priced_total",
    }


def _session_reference(records, source_revision: str | None):
    matching = [
        record
        for record in records
        if source_revision is not None and record.get("source_trial_revision") == source_revision
    ]
    selected = matching if matching else records if source_revision is None else []
    references = {}
    for record in selected:
        reference = record.get("session")
        if type(reference) is not dict:
            continue
        session_id = reference.get("session_id")
        path = reference.get("sqlite_path")
        if type(session_id) is not str or not session_id or len(session_id) > 2048:
            continue
        if path is not None and (
            type(path) is not str or len(path) > 4096 or not Path(path).is_absolute()
        ):
            continue
        references[session_id, path] = reference
    if len(references) != 1:
        return None, None, "unavailable"
    session_id, path = next(iter(references))
    return session_id, path, "result_revision" if matching else "provisional"


async def inspect_benchmark_campaign(
    directory: str | Path,
    *,
    include_sessions: bool = False,
    case_id: str | None = None,
    max_sessions: int = 100,
    max_diagnostics: int = 20,
    include_successors: bool = True,
) -> BenchmarkCampaignInspectionV1:
    """Observe every selected slot without importing or invoking the execution target."""

    if type(max_sessions) is not int or not 1 <= max_sessions <= 1000:
        raise ValueError("max_sessions must be between 1 and 1000.")
    if type(max_diagnostics) is not int or not 1 <= max_diagnostics <= 1000:
        raise ValueError("max_diagnostics must be between 1 and 1000.")
    root = Path(directory).resolve()
    campaign = load_benchmark_campaign(root)
    if case_id is not None and case_id not in {case.id for case in campaign.selection.cases}:
        raise ValueError("Requested case is not in the campaign cohort.")
    observations, limitations = _observations(root, campaign)
    store = (
        SQLiteEvalStore(root / "evals.sqlite3", read_only=True, schema_mode=SchemaMode.VALIDATE)
        if (root / "evals.sqlite3").is_file()
        else None
    )
    if store is None:
        limitations.add("eval_store_missing")
    rows = []
    trial_policy = None
    counts: Counter[str] = Counter()
    remaining_sessions, remaining_diagnostics = max_sessions, max_diagnostics
    try:
        for admitted in campaign.runs:
            record = None if store is None else await store.load_run(admitted.spec.id)
            native_links = (
                {}
                if record is None or store is None
                else {
                    (link.case_id, link.trial_number): link
                    for link in await store.load_trial_evidence_links(admitted.spec.id)
                }
            )
            if record is not None and record.spec != admitted.spec:
                raise ValueError("Stored campaign run conflicts with its exact admission receipt.")
            result = (
                None
                if record is None or record.status != "completed" or store is None
                else await store.load_result(admitted.spec.id)
            )
            if record is not None and record.status == "completed" and result is None:
                limitations.add("published_result_missing")
            if result is not None and (
                result.run.corpus_revision != admitted.spec.corpus_revision
                or result.run.suite_revision != admitted.spec.suite_revision
                or tuple(case.case_id for case in result.run.cases) != admitted.case_ids
            ):
                raise ValueError("Campaign result does not match the admitted cohort and corpus.")
            if result is not None:
                policy = result.run.trial_policy
                exposure = admitted.spec.invocation.authored_suite_exposure
                assert exposure is not None
                if policy.revision != exposure.trial_policy_revision:
                    raise ValueError("Published trial policy conflicts with campaign admission.")
                trial_policy = policy
            published = (
                {}
                if result is None
                else {
                    (case.case_id, trial.trial_number): trial
                    for case in result.run.cases
                    for trial in case.trials
                }
            )
            exposure = admitted.spec.invocation.authored_suite_exposure
            assert exposure is not None
            trials = exposure.candidate_trials // len(campaign.selection.cases)
            for selected_case in admitted.case_ids:
                for trial_number in range(1, trials + 1):
                    trial = published.get((selected_case, trial_number))
                    observed = observations.get((selected_case, trial_number), [])
                    status = "pending"
                    category = "pending"
                    code = None
                    if store is None:
                        status, category, code = (
                            "unavailable",
                            "evidence_unavailable",
                            "eval_store_missing",
                        )
                    if record is not None and record.status in {"failed", "cancelled"}:
                        status = "cancelled" if record.status == "cancelled" else "error"
                        category = (
                            "cancelled" if record.status == "cancelled" else "execution_failure"
                        )
                        code = (
                            str(record.failure_code)
                            if record.failure_diagnostic is None
                            else str(record.failure_diagnostic.reason)
                        )
                    if trial is not None:
                        status, category, code = (
                            trial.status,
                            benchmark_failure_category(trial),
                            trial.code.value,
                        )
                    counts[status] += 1
                    counts["total"] += 1
                    source_revision = None if trial is None else trial.source_trial_revision
                    session_id, session_path, association = _session_reference(
                        observed, source_revision
                    )
                    native_link = native_links.get((selected_case, trial_number))
                    if (
                        native_link is not None
                        and native_link.source_trial_revision == source_revision
                    ):
                        if session_id is not None and session_id != native_link.session_id:
                            raise ValueError(
                                "Observed session conflicts with the retained native trial checkpoint."
                            )
                        session_id, association = native_link.session_id, "result_revision"
                    row_limits = []
                    if association == "unavailable":
                        row_limits.append("exact_session_link_unavailable")
                    session_evidence = None
                    if (
                        include_sessions
                        and session_path is not None
                        and session_id is not None
                        and (case_id is None or selected_case == case_id)
                    ):
                        if remaining_sessions < 1 or remaining_diagnostics < 1:
                            row_limits.append("session_inspection_limit")
                        elif not Path(session_path).is_file():
                            row_limits.append("session_store_missing")
                        else:
                            session_store = None
                            try:
                                session_store = SQLiteSessionStore(
                                    session_path, read_only=True, schema_mode=SchemaMode.VALIDATE
                                )
                                session_evidence = await inspect_eval_sessions(
                                    session_store,
                                    session_id,
                                    max_sessions=remaining_sessions,
                                    max_diagnostics=remaining_diagnostics,
                                )
                                remaining_sessions -= max(1, len(session_evidence.sessions))
                                remaining_diagnostics -= len(session_evidence.diagnostics)
                            except (OSError, ValueError, RuntimeError) as exc:
                                row_limits.append(
                                    f"session_evidence_unavailable:{type(exc).__name__}"
                                )
                            finally:
                                if session_store is not None:
                                    await session_store.close()
                    rows.append(
                        BenchmarkTrialInspectionV1(
                            run_id=admitted.spec.id,
                            case_id=selected_case,
                            trial_number=trial_number,
                            run_status="not_admitted" if record is None else str(record.status),
                            result_state="published"
                            if trial is not None
                            else "provisional"
                            if observed
                            else "unavailable",
                            status=status,
                            failure_category=category,
                            diagnostic_code=code,
                            score=None if trial is None else trial.score,
                            source_trial_revision=source_revision,
                            output=None if trial is None else trial.output,
                            duration_ms=None if trial is None else trial.duration_ms,
                            usage=None
                            if trial is None or trial.usage_evidence_state == "unavailable"
                            else trial.usage,
                            usage_availability="unavailable"
                            if trial is None
                            or trial.usage is None
                            or trial.usage_evidence_state == "unavailable"
                            else "partial"
                            if trial.usage_evidence_state == "partial"
                            else "known",
                            **_cost_observations(trial),
                            evidence_state="unavailable"
                            if trial is None
                            else "complete"
                            if trial.evidence_complete
                            else "incomplete",
                            execution_status=None if trial is None else trial.execution_status,
                            capture_diagnostic=None
                            if trial is None or trial.capture_diagnostic is None
                            else trial.capture_diagnostic.model_dump(mode="json"),
                            failure_capture=None
                            if trial is None or trial.failure_capture is None
                            else trial.failure_capture.model_dump(mode="json"),
                            failure_evidence=None
                            if trial is None or trial.failure_evidence is None
                            else trial.failure_evidence.model_dump(mode="json"),
                            assertions=()
                            if trial is None
                            else tuple(_assertion_summary(item) for item in trial.assertions),
                            session_id=session_id,
                            session_sqlite_path=session_path,
                            session_association=association,
                            session_evidence=session_evidence,
                            prior_session_references=_prior_session_references(
                                observed, source_revision
                            ),
                            limitations=tuple(row_limits),
                            dashboard_path="/cayu/evals?"
                            + urlencode(
                                {
                                    "tab": "runs",
                                    "target": admitted.spec.target_key,
                                    "run": admitted.spec.id,
                                }
                            ),
                        )
                    )
    finally:
        if store is not None:
            await store.close()
    status = _campaign_status(rows, trial_policy)
    inspection = BenchmarkCampaignInspectionV1(
        campaign=campaign,
        trial_policy=trial_policy,
        status=status,
        counts=dict(counts),
        trials=tuple(rows),
        limitations=tuple(sorted(limitations)),
    )
    if include_successors and campaign.retry_of is None:
        successors, successor_limits = await _successor_summaries(root, inspection)
        inspection = inspection.model_copy(
            update={
                "successors": successors,
                "limitations": tuple(sorted(set(inspection.limitations) | successor_limits)),
            }
        )
    if len(inspection.model_dump_json().encode("utf-8")) > BENCHMARK_INSPECTION_MAX_BYTES:
        raise ValueError("Benchmark inspection exceeds the bounded export size.")
    return inspection


def _prior_session_references(records, published_revision):
    references = []
    for record in records:
        if (
            record.get("source_trial_revision") == published_revision
            and published_revision is not None
        ):
            continue
        session_id, path, _association = _session_reference([record], None)
        if session_id is None:
            continue
        reference = {
            "session_id": session_id,
            "sqlite_path": path,
            "association": "provisional",
            "state": record.get("state", "unknown"),
        }
        if reference not in references:
            references.append(reference)
        if len(references) == 10:
            break
    return tuple(references)


async def _successor_summaries(root: Path, parent: BenchmarkCampaignInspectionV1):
    retry_root = root / "retries"
    if not retry_root.exists():
        return (), set()
    if retry_root.is_symlink():
        raise ValueError("Retry evidence directories cannot be symbolic links.")
    slots = {(row.case_id, row.trial_number): row for row in parent.trials}
    rows = []
    limits = set()
    for index, slot_directory in enumerate(retry_root.iterdir()):
        if index >= 32:
            limits.add("successor_inspection_limit")
            break
        if slot_directory.is_symlink() or not slot_directory.is_dir():
            limits.add("invalid_successor_directory")
            continue
        for attempt in range(1, parent.campaign.settings.max_retry_attempts + 1):
            directory = slot_directory / str(attempt)
            if not (directory / "campaign.json").is_file():
                continue
            if directory.is_symlink():
                raise ValueError("Retry evidence directories cannot be symbolic links.")
            child = await inspect_benchmark_campaign(directory, include_successors=False)
            link = child.campaign.retry_of
            if link is None:
                raise ValueError("Successor evidence has no original-trial lineage.")
            original = slots.get((link.case_id, link.trial_number))
            expected_slot = _content_revision(
                {
                    "campaign": parent.campaign.revision,
                    "case": link.case_id,
                    "trial": link.trial_number,
                },
                "benchmark retry slot",
            )[7:]
            if (
                link.campaign_revision != parent.campaign.revision
                or link.attempt != attempt
                or expected_slot != slot_directory.name
                or original is None
                or link.source_trial_revision != original.source_trial_revision
                or link.run_id != original.run_id
            ):
                raise ValueError("Successor evidence conflicts with its original-trial identity.")
            rows.append(
                {
                    "directory": str(directory.relative_to(root)),
                    "campaign_revision": child.campaign.revision,
                    "retry_of": link.model_dump(mode="json"),
                    "status": child.status,
                    "counts": child.counts,
                    "observed_total_tokens": str(
                        sum(
                            trial.usage.total_tokens
                            for trial in child.trials
                            if trial.usage is not None
                        )
                    )
                    if any(trial.usage is not None for trial in child.trials)
                    else None,
                    "trials_with_unavailable_usage": sum(
                        trial.usage is None for trial in child.trials
                    ),
                    "cost_availability": child.trials[0].cost_availability,
                    "costs": child.trials[0].costs,
                }
            )
    return tuple(rows), limits


def render_benchmark_campaign_html(inspection: BenchmarkCampaignInspectionV1) -> str:
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                trial.case_id,
                trial.trial_number,
                trial.status,
                trial.score if trial.score is not None else "unavailable",
                trial.failure_category,
                trial.evidence_state,
                trial.usage_availability,
                trial.session_id or "unavailable",
            )
        )
        + f"<td><a href='{html.escape(trial.dashboard_path, quote=True)}'>Open run</a></td></tr>"
        for trial in inspection.trials
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Benchmark campaign</title>"
        "<style>body{font:16px system-ui;margin:2rem;color:#17202a}table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        f"<h1>{html.escape(inspection.campaign.package_id)} · {html.escape(inspection.status)}</h1>"
        f"<p>{html.escape(inspection.campaign.id)}</p><table><thead><tr>"
        "<th>Case</th><th>Trial</th><th>Status</th><th>Score</th><th>Category</th><th>Evidence</th><th>Usage</th><th>Session</th><th>Dashboard</th>"
        f"</tr></thead><tbody>{rows}</tbody></table><details><summary>Complete inspection</summary><pre>"
        + html.escape(inspection.model_dump_json(indent=2))
        + "</pre></details></html>"
    )


async def export_benchmark_campaign(
    directory: str | Path, output: str | Path
) -> BenchmarkCampaignInspectionV1:
    """Export bounded outcome/identity views; never sweep stores, logs, or package truth."""

    root, destination = Path(directory).resolve(), Path(output).resolve()
    if destination.is_relative_to(root):
        raise ValueError("Benchmark export must be outside the campaign directory.")
    inspection = await inspect_benchmark_campaign(root)
    content = inspection.model_dump_json(indent=2).encode("utf-8")
    export = json.dumps(
        {
            "schema_version": 1,
            "campaign_id": inspection.campaign.id,
            "files": [
                {
                    "path": "inspection.json",
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    with destination.open("xb") as handle:
        destination.chmod(0o600)
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("inspection.json", content)
            archive.writestr("export.json", export)
    return inspection


def load_benchmark_export(path: str | Path) -> BenchmarkCampaignInspectionV1:
    """Validate an offline receipt without opening any recorded source locator."""

    raw = read_regular_file(Path(path), max_bytes=BENCHMARK_INSPECTION_MAX_BYTES + 1024 * 1024)
    if raw is None:
        raise ValueError("Benchmark export is missing.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 2 or {member.filename for member in members} != {
            "export.json",
            "inspection.json",
        }:
            raise ValueError("Benchmark export has unexpected archive members.")
        if any(
            member.file_size
            > (16_384 if member.filename == "export.json" else BENCHMARK_INSPECTION_MAX_BYTES)
            for member in members
        ):
            raise ValueError("Benchmark export exceeds its bounded member size.")
        manifest = decode_document(archive.read("export.json"))
        content = archive.read("inspection.json")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("files")
        != [
            {
                "path": "inspection.json",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ]
    ):
        raise ValueError("Benchmark export hash or manifest is invalid.")
    decode_document(content)  # Reject duplicate JSON members before typed JSON decoding.
    inspection = BenchmarkCampaignInspectionV1.model_validate_json(content)
    if manifest.get("campaign_id") != inspection.campaign.id:
        raise ValueError("Benchmark export identity does not match its inspection.")
    return inspection.model_copy(
        update={
            "limitations": tuple(
                sorted(set(inspection.limitations) | {"offline_export", "linked_stores_not_opened"})
            )
        }
    )
