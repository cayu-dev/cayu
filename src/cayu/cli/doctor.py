from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.cli.check import build_project_check_report
from cayu.cli.project import (
    build_project_app,
    build_project_service,
    project_context,
    resolve_project,
)
from cayu.cli.project_control_plane import (
    build_project_control_plane_context,
    close_project_control_plane_context,
)
from cayu.cli.scaffold_check import check_declared_scaffold_source
from cayu.project_control_plane import resolve_project_control_plane_context
from cayu.runtime.checks import DiagnosticSeverity, ProjectControlPlaneCheckEvidence
from cayu.runtime.sessions import MAX_SESSION_ID_BYTES
from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.support_bundles import (
    DEFAULT_SUPPORT_BUNDLE_LIMITS,
    SUPPORT_BUNDLE_SCHEMA_VERSION,
    CollectorDisposition,
    SupportBundleContext,
    SupportBundleOutcome,
    SupportBundleReport,
    SupportCollectorResult,
    builtin_support_collectors,
    cleanup_support_bundle_staging,
    collect_support_bundle,
    encode_support_bundle,
    minimal_support_bundle_report,
    support_bundle_staging_name,
    write_support_bundle_atomic,
)

_EXIT_BY_OUTCOME = {
    SupportBundleOutcome.CLEAN: 0,
    SupportBundleOutcome.PARTIAL: 1,
    SupportBundleOutcome.BOOT_FAILED: 2,
    SupportBundleOutcome.VALIDATION_FAILED: 3,
}
_PROCESS_TERMINATE_SECONDS = 2.0
_PROCESS_KILL_SECONDS = 2.0
_PROCESS_TEARDOWN_RESERVE_SECONDS = (
    DEFAULT_SUPPORT_BUNDLE_LIMITS.reconciliation_timeout_seconds
    + _PROCESS_TERMINATE_SECONDS
    + _PROCESS_KILL_SECONDS
    + 1.0
)
_SELF_DEADLINE_EXIT_CODE = 124


def add_doctor_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="Write a bounded, redacted diagnostic support bundle.",
        description=(
            "Boot the selected Cayu project without running agent work and atomically "
            "write a bounded, redacted diagnostic support bundle."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        metavar="OUTPUT",
        help="Output ZIP path. The file is written atomically with restrictive permissions.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        metavar="SESSION_ID",
        help=(
            "Include a bounded envelope-only durable event tail for this session "
            "(repeatable; no history is read by default)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the stable command outcome as JSON.",
    )


def run_doctor(args: argparse.Namespace) -> int:
    with _sigterm_cleanup_boundary():
        return _run_doctor_before_deadline(args)


def _run_doctor_before_deadline(args: argparse.Namespace) -> int:
    command_deadline = time.monotonic() + DEFAULT_SUPPORT_BUNDLE_LIMITS.command_timeout_seconds
    sessions = tuple(args.session)
    request_error = _validate_session_selectors(sessions)
    if request_error is not None:
        report = minimal_support_bundle_report(
            outcome=SupportBundleOutcome.VALIDATION_FAILED,
            reason_code=request_error,
        )
    else:
        worker_budget = min(
            DEFAULT_SUPPORT_BUNDLE_LIMITS.worker_timeout_seconds,
            max(
                0.0,
                command_deadline
                - time.monotonic()
                - DEFAULT_SUPPORT_BUNDLE_LIMITS.publication_timeout_seconds
                - _PROCESS_TEARDOWN_RESERVE_SECONDS,
            ),
        )
        report = (
            minimal_support_bundle_report(
                outcome=SupportBundleOutcome.BOOT_FAILED,
                reason_code="command_deadline_elapsed",
            )
            if worker_budget <= 0
            else _run_bounded_worker(
                args.target,
                sessions,
                worker_timeout_seconds=worker_budget,
            )
        )

    try:
        archive = encode_support_bundle(report)
    except Exception:
        report = minimal_support_bundle_report(
            outcome=SupportBundleOutcome.VALIDATION_FAILED,
            reason_code="bundle_validation_failed",
        )
        try:
            archive = encode_support_bundle(report)
        except Exception:
            _render_doctor_result(
                outcome=SupportBundleOutcome.VALIDATION_FAILED,
                as_json=args.json,
                written=False,
            )
            return 3

    publication_budget = min(
        DEFAULT_SUPPORT_BUNDLE_LIMITS.publication_timeout_seconds,
        max(
            0.0,
            command_deadline - time.monotonic() - _PROCESS_TEARDOWN_RESERVE_SECONDS,
        ),
    )
    if publication_budget <= 0 or not _run_bounded_publisher(
        args.bundle,
        archive,
        publication_timeout_seconds=publication_budget,
        command_deadline=command_deadline,
    ):
        _render_output_write_failure(as_json=args.json)
        return 4

    _render_doctor_result(outcome=report.outcome, as_json=args.json, written=True)
    return _EXIT_BY_OUTCOME[report.outcome]


@contextmanager
def _sigterm_cleanup_boundary():
    """Translate SIGTERM into a supervisory exit after owned cleanup runs."""

    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(sigterm)

    def terminate(signum: int, _frame: object) -> None:
        # Ignore repeated termination while the first signal unwinds through
        # process cleanup. The prior handler is restored by the owner below.
        signal.signal(sigterm, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(sigterm, terminate)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


def _bounded_process_entry(
    lifetime_seconds: float,
    entry: Callable[..., None],
    *args: object,
) -> None:
    """Give an owned child a deadline that survives loss of its parent."""

    def expire() -> None:
        os._exit(_SELF_DEADLINE_EXIT_CODE)

    watchdog = threading.Timer(lifetime_seconds, expire)
    watchdog.daemon = True
    watchdog.start()
    # Leave the daemon watchdog armed after the entry returns. Project code can
    # start a non-daemon thread that holds the child in interpreter shutdown;
    # the process lifetime, not only the target call, is the bounded resource.
    entry(*args)


def _bounded_process_join_timeout(
    timeout_seconds: float,
    *,
    deadline: float | None,
    reserve_seconds: float = 0.0,
) -> float:
    if deadline is None:
        return timeout_seconds
    return min(
        timeout_seconds,
        max(0.0, deadline - time.monotonic() - reserve_seconds),
    )


def _terminate_owned_process(
    process: BaseProcess,
    *,
    deadline: float | None = None,
) -> bool:
    if process.is_alive():
        process.terminate()
        process.join(
            timeout=_bounded_process_join_timeout(
                _PROCESS_TERMINATE_SECONDS,
                deadline=deadline,
                reserve_seconds=_PROCESS_KILL_SECONDS,
            )
        )
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(
            timeout=_bounded_process_join_timeout(
                _PROCESS_KILL_SECONDS,
                deadline=deadline,
            )
        )
    return not process.is_alive()


def _process_was_started(process: BaseProcess) -> bool:
    try:
        return process.pid is not None
    except ValueError:
        return False


def _run_bounded_publisher(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    publication_timeout_seconds: float,
    command_deadline: float | None = None,
    _publisher_entry: Callable[[str, bytes, str], None] | None = None,
    _reconciliation_timeout_seconds: float | None = None,
    _reconciliation_entry: Callable[[str, str], None] | None = None,
) -> bool:
    """Publish in one owned process and accept only a settled successful exit."""

    publication_deadline = time.monotonic() + publication_timeout_seconds
    if command_deadline is not None:
        publication_deadline = min(publication_deadline, command_deadline)
    context = multiprocessing.get_context("spawn")
    entry = _doctor_publisher_entry if _publisher_entry is None else _publisher_entry
    try:
        temporary_name = support_bundle_staging_name(path)
    except (OSError, ValueError):
        return False
    process = context.Process(
        target=_bounded_process_entry,
        args=(
            publication_timeout_seconds,
            entry,
            os.fspath(path),
            payload,
            temporary_name,
        ),
        name="cayu-doctor-publisher",
    )
    started = False
    settled = False
    deadline_elapsed = False
    exitcode: int | None = None
    published = False
    with _sigterm_cleanup_boundary():
        try:
            try:
                process.start()
                started = True
                process.join(timeout=max(0.0, publication_deadline - time.monotonic()))
                deadline_elapsed = time.monotonic() >= publication_deadline
            except Exception:
                pass
        finally:
            started = started or _process_was_started(process)
            if started:
                settled = _terminate_owned_process(process, deadline=command_deadline)
                exitcode = process.exitcode
                if settled:
                    with suppress(ValueError):
                        process.close()
            published = not deadline_elapsed and settled and exitcode == 0
            # Never race reconciliation against a publisher whose quiescence
            # could not be proved. Its independent lifetime guard remains the
            # only safe owner in that exceptional state.
            if not published and settled:
                _run_bounded_staging_reconciliation(
                    path,
                    temporary_name,
                    timeout_seconds=(
                        DEFAULT_SUPPORT_BUNDLE_LIMITS.reconciliation_timeout_seconds
                        if _reconciliation_timeout_seconds is None
                        else _reconciliation_timeout_seconds
                    ),
                    command_deadline=command_deadline,
                    _reconciliation_entry=_reconciliation_entry,
                )
    return published


def _doctor_publisher_entry(path: str, payload: bytes, temporary_name: str) -> None:
    _silence_worker_side_channels()
    try:
        write_support_bundle_atomic(path, payload, _temporary_name=temporary_name)
    except BaseException:
        raise SystemExit(1) from None


def _run_bounded_staging_reconciliation(
    path: str | os.PathLike[str],
    temporary_name: str,
    *,
    timeout_seconds: float,
    command_deadline: float | None = None,
    _reconciliation_entry: Callable[[str, str], None] | None = None,
) -> bool:
    """Reconcile one staging leaf without blocking the command indefinitely."""

    reconciliation_deadline = time.monotonic() + timeout_seconds
    if command_deadline is not None:
        reconciliation_deadline = min(reconciliation_deadline, command_deadline)
        if reconciliation_deadline <= time.monotonic():
            return False
    context = multiprocessing.get_context("spawn")
    entry = (
        _doctor_staging_reconciliation_entry
        if _reconciliation_entry is None
        else _reconciliation_entry
    )
    process = context.Process(
        target=_bounded_process_entry,
        args=(timeout_seconds, entry, os.fspath(path), temporary_name),
        name="cayu-doctor-reconciliation",
    )
    started = False
    settled = False
    deadline_elapsed = False
    exitcode: int | None = None
    try:
        try:
            process.start()
            started = True
        except Exception:
            return False
        process.join(timeout=max(0.0, reconciliation_deadline - time.monotonic()))
        deadline_elapsed = time.monotonic() >= reconciliation_deadline
    finally:
        started = started or _process_was_started(process)
        if started:
            settled = _terminate_owned_process(process, deadline=command_deadline)
            exitcode = process.exitcode
            if settled:
                with suppress(ValueError):
                    process.close()
    return not deadline_elapsed and settled and exitcode == 0


def _doctor_staging_reconciliation_entry(path: str, temporary_name: str) -> None:
    _silence_worker_side_channels()
    try:
        cleanup_support_bundle_staging(path, temporary_name)
    except BaseException:
        raise SystemExit(1) from None


def _validate_session_selectors(sessions: tuple[str, ...]) -> str | None:
    if len(sessions) > DEFAULT_SUPPORT_BUNDLE_LIMITS.max_sessions:
        return "too_many_session_selectors"
    if any(type(item) is not str for item in sessions):
        return "invalid_session_selector"
    for item in sessions:
        try:
            require_clean_nonblank(require_unicode_scalar_text(item, "session_id"), "session_id")
        except ValueError:
            return "invalid_session_selector"
        if len(item.encode("utf-8")) > MAX_SESSION_ID_BYTES:
            return "session_selector_too_large"
    if len(set(sessions)) != len(sessions):
        return "duplicate_session_selector"
    return None


def _run_bounded_worker(
    target: str | None,
    sessions: tuple[str, ...],
    *,
    worker_timeout_seconds: float = (DEFAULT_SUPPORT_BUNDLE_LIMITS.worker_timeout_seconds),
    _worker_entry: Callable[[Connection, str | None, tuple[str, ...]], None] | None = None,
) -> SupportBundleReport:
    worker_deadline = time.monotonic() + worker_timeout_seconds
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_bounded_process_entry,
        args=(
            worker_timeout_seconds,
            _doctor_worker_entry if _worker_entry is None else _worker_entry,
            child_connection,
            target,
            sessions,
        ),
        name="cayu-doctor",
    )
    received: list[bytes | None] = []
    receiver: threading.Thread | None = None
    deadline_elapsed = True
    process_started = False
    process_settled = False
    process_exitcode: int | None = None
    with _sigterm_cleanup_boundary():
        try:
            try:
                process.start()
                process_started = True
            except Exception:
                return minimal_support_bundle_report(
                    outcome=SupportBundleOutcome.BOOT_FAILED,
                    reason_code="worker_start_failed",
                )
            child_connection.close()

            def receive() -> None:
                try:
                    payload = parent_connection.recv_bytes(
                        maxlength=DEFAULT_SUPPORT_BUNDLE_LIMITS.max_bundle_bytes
                    )
                except BaseException:
                    payload = None
                received.append(payload)

            receiver = threading.Thread(
                target=receive,
                name="cayu-doctor-result",
                daemon=True,
            )
            receiver.start()
            receiver.join(timeout=max(0.0, worker_deadline - time.monotonic()))
            deadline_elapsed = receiver.is_alive() or time.monotonic() >= worker_deadline
            if not deadline_elapsed:
                process.join(timeout=max(0.0, worker_deadline - time.monotonic()))
                deadline_elapsed = process.is_alive() or time.monotonic() >= worker_deadline
        finally:
            child_connection.close()
            process_started = process_started or _process_was_started(process)
            if process_started:
                process_settled = _terminate_owned_process(process)
                process_exitcode = process.exitcode
            parent_connection.close()
            if receiver is not None and receiver.is_alive():
                receiver.join(timeout=1.0)
            if process_started and process_settled:
                with suppress(ValueError):
                    process.close()

    payload = (
        None
        if deadline_elapsed or not process_settled or process_exitcode != 0 or not received
        else received[0]
    )

    if payload is None:
        return minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="worker_deadline_or_exit",
        )
    try:
        return SupportBundleReport.model_validate_json(payload)
    except Exception:
        return minimal_support_bundle_report(
            outcome=SupportBundleOutcome.VALIDATION_FAILED,
            reason_code="worker_payload_invalid",
        )


def _doctor_worker_entry(
    connection: Connection,
    target: str | None,
    sessions: tuple[str, ...],
) -> None:
    _silence_worker_side_channels()
    try:
        report = _collect_project_report(target=target, sessions=sessions)
        payload = report.model_dump_json().encode("utf-8")
        if len(payload) > DEFAULT_SUPPORT_BUNDLE_LIMITS.max_bundle_bytes:
            report = minimal_support_bundle_report(
                outcome=SupportBundleOutcome.VALIDATION_FAILED,
                reason_code="worker_payload_too_large",
            )
            payload = report.model_dump_json().encode("utf-8")
        connection.send_bytes(payload)
    except BaseException:
        try:
            fallback = minimal_support_bundle_report(
                outcome=SupportBundleOutcome.BOOT_FAILED,
                reason_code="worker_failed",
            )
            connection.send_bytes(fallback.model_dump_json().encode("utf-8"))
        except BaseException:
            pass
    finally:
        connection.close()


def _silence_worker_side_channels() -> None:
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor not in {1, 2}:
            os.close(descriptor)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def _collect_project_report(
    *,
    target: str | None,
    sessions: tuple[str, ...],
) -> SupportBundleReport:
    with diagnostic_store_inspection() as inspection:
        report = _collect_project_report_under_inspection(
            target=target,
            sessions=sessions,
        )
        try:
            inspection.verify()
        except Exception:
            report = _with_store_inspection_failure(report)
    return report


def _collect_project_report_under_inspection(
    *,
    target: str | None,
    sessions: tuple[str, ...],
) -> SupportBundleReport:
    control_plane_context = None
    report: SupportBundleReport | None = None
    stage = "project_discovery"
    try:
        project = resolve_project(target, command="cayu doctor")
        stage = "source_validation"
        source_diagnostics = check_declared_scaffold_source(
            project.root,
            tags=frozenset(),
            deploy_only=False,
        )
        if any(item.severity is DiagnosticSeverity.ERROR for item in source_diagnostics):
            return minimal_support_bundle_report(
                outcome=SupportBundleOutcome.BOOT_FAILED,
                reason_code="source_validation_failed",
            )

        stage = "control_plane_boot"
        control_plane_context = build_project_control_plane_context(
            project.root,
            mode="production",
        )
        stage = "application_boot"
        with project_context(project.root):
            service = (
                None
                if project.service_target is None
                else build_project_service(
                    project.service_target,
                    mode="production",
                    command="Doctor",
                    project_context=control_plane_context,
                )
            )
            app = (
                build_project_app(project.target, command="Doctor")
                if service is None
                else service.cayu_app
            )
            manifest = app.describe(project_root=project.root)
            control_plane_diagnostics = (
                None if service is None else service._support_bundle_system_diagnostics()
            )

        stage = "check_collection"
        check_evidence = ProjectControlPlaneCheckEvidence(
            project_identity_configured=(control_plane_context.project_identity_configured),
            eval_store_configured=control_plane_context.eval_store_configured,
            service_context=(
                "not_applicable"
                if service is None
                else (
                    "attached"
                    if service.project_control_plane_context_attached
                    else "migration_required"
                )
            ),
        )
        check_report = build_project_check_report(
            project.root,
            manifest,
            service_manifest=None if service is None else service.manifest,
            project_control_plane=check_evidence,
        )
        resolved_context = resolve_project_control_plane_context(
            control_plane_context,
            app,
        )
        if resolved_context is None:
            raise RuntimeError("project control-plane context was not resolved.")
        summary = resolved_context.safe_summary()
        eval_summary = summary["eval_store"]

        stage = "collector_execution"
        report = asyncio.run(
            collect_support_bundle(
                SupportBundleContext(
                    app=app,
                    manifest=manifest,
                    check_report=check_report,
                    service_manifest=None if service is None else service.manifest,
                    project_id=summary["project_id"],
                    application_release_id=summary["application_release_id"],
                    eval_backend=eval_summary["backend"],
                    eval_source=eval_summary["source"],
                    eval_store=resolved_context.eval_store,
                    control_plane_diagnostics=control_plane_diagnostics,
                ),
                builtin_support_collectors(session_selectors=sessions),
            )
        )
    except BaseException:
        report = minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code=f"{stage}_failed",
        )
    finally:
        if control_plane_context is not None:
            try:
                close_project_control_plane_context(control_plane_context)
            except BaseException:
                if report is None:
                    report = minimal_support_bundle_report(
                        outcome=SupportBundleOutcome.BOOT_FAILED,
                        reason_code=f"{stage}_failed",
                    )
                report = _with_cleanup_failure(report)

    if report is None:
        return minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="collection_failed",
        )
    return report


def _with_store_inspection_failure(
    report: SupportBundleReport,
) -> SupportBundleReport:
    return _with_report_failure(
        report,
        name="store_inspection",
        reason_code="store_inspection_changed",
        outcome=SupportBundleOutcome.BOOT_FAILED,
    )


def _with_cleanup_failure(report: SupportBundleReport) -> SupportBundleReport:
    return _with_report_failure(
        report,
        name="control_plane_cleanup",
        reason_code="control_plane_cleanup_failed",
        outcome=(
            SupportBundleOutcome.BOOT_FAILED
            if report.outcome is SupportBundleOutcome.BOOT_FAILED
            else SupportBundleOutcome.PARTIAL
        ),
    )


def _with_report_failure(
    report: SupportBundleReport,
    *,
    name: str,
    reason_code: str,
    outcome: SupportBundleOutcome,
) -> SupportBundleReport:
    failure = SupportCollectorResult(
        name=name,
        disposition=CollectorDisposition.FAILED,
        duration_ms=0,
        evidence_bytes=0,
        reason_code=reason_code,
    )
    return SupportBundleReport.from_results(
        bundle_id=report.bundle_id,
        generated_at=report.generated_at,
        outcome=outcome,
        limits=report.limits,
        collection_duration_ms=report.collection_duration_ms,
        collectors=(*report.collectors, failure),
    )


def _render_doctor_result(
    *,
    outcome: SupportBundleOutcome,
    as_json: bool,
    written: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "bundle_written": written,
                    "outcome": outcome.value,
                    "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"Diagnostic support bundle: {outcome.value}.")


def _render_output_write_failure(*, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "bundle_written": False,
                    "outcome": "output_write_failed",
                    "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
        )
    else:
        print("Diagnostic support bundle: output_write_failed.", file=sys.stderr)
