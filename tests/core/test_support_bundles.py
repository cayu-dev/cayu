from __future__ import annotations

import asyncio
import io
import json
import os
import stat
import threading
import time
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

import cayu.runtime.sessions as sessions_runtime
import cayu.runtime.tasks as tasks_runtime
import cayu.support_bundles as support_bundles
from cayu import (
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    InMemoryTaskStore,
    LocalArtifactStore,
    McpManifestPolicy,
    McpServerSpec,
    RecoveryCleanupDeadlineScope,
    RecoveryCleanupPolicy,
    RecoveryCleanupRetainedTaskSnapshot,
    RecoveryCleanupSupervisorSnapshot,
    RunRequest,
    SecretRedactor,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
)
from cayu.runtime.checks import check_manifest
from cayu.runtime.sessions import InMemorySessionStore, SessionIdentity
from cayu.support_bundles import (
    ArtifactAvailabilityEvidence,
    CollectorDisposition,
    FunctionalSupportBundleCollector,
    ManifestSummaryEvidence,
    RecoveryCleanupEvidence,
    RecoveryCleanupPolicyEvidence,
    RecoveryCleanupRetainedEvidence,
    RuntimeIdentityEvidence,
    SessionEventTailEvidence,
    StoreSummaryEvidence,
    SupportBundleContext,
    SupportBundleLimits,
    SupportBundleOutcome,
    SupportBundleReport,
    SupportCollectorOutput,
    SupportCollectorResult,
    TaskOperationalEvidence,
    builtin_support_collectors,
    collect_support_bundle,
    collected,
    encode_support_bundle,
    minimal_support_bundle_report,
    unavailable,
    validate_support_bundle_archive,
    write_support_bundle_atomic,
)


def _context(
    *,
    app: CayuApp | None = None,
    limits: SupportBundleLimits | None = None,
) -> SupportBundleContext:
    app = app or CayuApp(enable_logging=False)
    manifest = app.describe()
    return SupportBundleContext(
        app=app,
        manifest=manifest,
        check_report=check_manifest(manifest),
        service_manifest=None,
        project_id=None,
        application_release_id=f"manifest-{manifest.fingerprint}",
        eval_backend=None,
        eval_source=None,
        limits=limits
        or SupportBundleLimits(
            collector_timeout_seconds=1,
            collection_timeout_seconds=5,
            worker_timeout_seconds=10,
            publication_timeout_seconds=5,
            reconciliation_timeout_seconds=2,
            command_timeout_seconds=20,
            max_items=100,
            max_sessions=10,
            event_limit=50,
            event_query_bytes=256 * 1024,
            max_collector_bytes=256 * 1024,
            max_evidence_bytes=1024 * 1024,
            max_bundle_bytes=2 * 1024 * 1024,
        ),
    )


def _runtime_evidence(value: str = "test") -> RuntimeIdentityEvidence:
    return RuntimeIdentityEvidence(
        cayu_version=value,
        python_version="3.11.0",
        python_implementation="cpython",
        operating_system="TestOS",
        machine="test-machine",
    )


def test_archive_has_only_typed_report_and_derived_summary(tmp_path: Path) -> None:
    evidence = _runtime_evidence()
    evidence_bytes = len(
        json.dumps(
            evidence.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    report = SupportBundleReport.from_results(
        generated_at=datetime(2026, 1, 2, tzinfo=UTC),
        outcome=SupportBundleOutcome.PARTIAL,
        limits=_context().limits,
        collection_duration_ms=3,
        collectors=(
            SupportCollectorResult(
                name="runtime",
                disposition=CollectorDisposition.COLLECTED,
                duration_ms=2,
                evidence_bytes=evidence_bytes,
                evidence=evidence,
            ),
            SupportCollectorResult(
                name="optional",
                disposition=CollectorDisposition.UNAVAILABLE,
                duration_ms=1,
                evidence_bytes=0,
                reason_code="not_supported",
            ),
        ),
    )

    assert report.evidence_complete is False
    with pytest.raises(ValueError, match="clean outcome cannot contain non-collected results"):
        SupportBundleReport.from_results(
            generated_at=report.generated_at,
            outcome=SupportBundleOutcome.CLEAN,
            limits=report.limits,
            collection_duration_ms=report.collection_duration_ms,
            collectors=report.collectors,
        )

    payload = encode_support_bundle(report)
    validated = validate_support_bundle_archive(payload)
    assert validated == report
    with zipfile.ZipFile(Path(tmp_path, "copy.zip"), mode="w") as _unused:
        pass
    with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
        assert archive.namelist() == ["report.json", "summary.txt"]
        document = json.loads(archive.read("report.json"))
        assert document["schema_version"] == "1"
        assert document["command_version"] == "1"
        assert document["outcome"] == "partial"
        assert document["bundle_id"].startswith("bundle_")
        assert document["collection_duration_ms"] == 3
        assert document["collector_count"] == 2
        assert document["collected_count"] == 1
        assert document["omitted_count"] == 1
        assert (
            archive.read("summary.txt")
            .decode()
            .endswith("- optional: unavailable (not_supported); 1 ms, 0 bytes\n")
        )


def test_runner_reports_clean_only_when_every_collector_is_collected() -> None:
    async def succeed(_context):
        return collected(_runtime_evidence())

    report = asyncio.run(
        collect_support_bundle(
            _context(),
            (FunctionalSupportBundleCollector("success", succeed),),
        )
    )

    assert report.outcome is SupportBundleOutcome.CLEAN
    assert report.evidence_complete is True
    assert report.omitted_count == 0


def test_runner_preserves_successes_and_types_failures_and_unavailability() -> None:
    async def succeed(_context):
        return collected(_runtime_evidence())

    async def unsupported(_context):
        return unavailable("typed_projection_missing")

    async def fail(_context):
        raise RuntimeError("raw-secret-canary")

    async def time_out(_context):
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    limits = _context().limits.model_copy(
        update={
            "collector_timeout_seconds": 0.01,
            "collection_timeout_seconds": 1,
        }
    )
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (
                FunctionalSupportBundleCollector("success", succeed),
                FunctionalSupportBundleCollector("unsupported", unsupported),
                FunctionalSupportBundleCollector("failure", fail),
                FunctionalSupportBundleCollector("timeout", time_out),
            ),
            now=lambda: datetime(2026, 1, 2, tzinfo=UTC),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [
        (item.name, item.disposition.value, item.reason_code) for item in report.collectors
    ] == [
        ("success", "collected", None),
        ("unsupported", "unavailable", "typed_projection_missing"),
        ("failure", "failed", "collector_failed"),
        ("timeout", "timed_out", "collector_deadline_elapsed"),
    ]
    assert "raw-secret-canary" not in report.model_dump_json()


def test_runner_does_not_classify_prompt_timeout_error_as_its_deadline() -> None:
    async def prompt_timeout(_context) -> SupportCollectorOutput:
        raise TimeoutError("collector-timeout-secret-canary")

    async def succeed(_context) -> SupportCollectorOutput:
        return collected(_runtime_evidence("later"))

    report = asyncio.run(
        collect_support_bundle(
            _context(),
            (
                FunctionalSupportBundleCollector("prompt_timeout", prompt_timeout),
                FunctionalSupportBundleCollector("later", succeed),
            ),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [(item.disposition, item.reason_code) for item in report.collectors] == [
        (CollectorDisposition.FAILED, "collector_failed"),
        (CollectorDisposition.COLLECTED, None),
    ]
    assert "collector-timeout-secret-canary" not in report.model_dump_json()


def test_runner_uses_real_task_cancellation_for_deadline() -> None:
    cancellation_counts: list[int] = []

    async def time_out(_context) -> SupportCollectorOutput:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            assert task is not None
            cancellation_counts.append(task.cancelling())
            raise
        raise AssertionError("unset event unexpectedly completed")

    limits = _context().limits.model_copy(
        update={
            "collector_timeout_seconds": 0.01,
            "collection_timeout_seconds": 1,
        }
    )
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (FunctionalSupportBundleCollector("timeout", time_out),),
        )
    )

    assert cancellation_counts and cancellation_counts[0] >= 1
    assert report.collectors[0].disposition is CollectorDisposition.TIMED_OUT


@pytest.mark.parametrize("collector_name", ("sessions", "tasks"))
def test_in_memory_operational_collectors_cooperate_with_deadline_and_preserve_results(
    collector_name: str,
    monkeypatch,
) -> None:
    async def scenario() -> tuple[SupportBundleReport, bool, list[int]]:
        if collector_name == "sessions":
            session_store = InMemorySessionStore()
            for index in range(sessions_runtime._IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL):
                await session_store.create(
                    RunRequest(
                        session_id=f"session-{index}",
                        agent_name="assistant",
                        environment_name="local",
                        messages=[],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="fake"),
                )
            app = CayuApp(session_store=session_store, enable_logging=False)
            owner_module = sessions_runtime
        else:
            task_store = InMemoryTaskStore()
            for index in range(tasks_runtime._IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL):
                await task_store.create_task(TaskCreate(task_id=f"task-{index}", type="test"))
            app = CayuApp(task_store=task_store, enable_logging=False)
            owner_module = tasks_runtime

        checkpoint_started = asyncio.Event()
        never_complete = asyncio.Event()
        cancellation_counts: list[int] = []

        async def block_at_checkpoint() -> None:
            checkpoint_started.set()
            try:
                await never_complete.wait()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                assert task is not None
                cancellation_counts.append(task.cancelling())
                raise

        monkeypatch.setattr(
            owner_module,
            "_cooperate_with_in_memory_aggregate_cancellation",
            block_at_checkpoint,
        )

        async def succeed(_context) -> SupportCollectorOutput:
            return collected(_runtime_evidence())

        operational = next(
            collector
            for collector in builtin_support_collectors()
            if collector.name == collector_name
        )
        limits = _context(app=app).limits.model_copy(
            update={
                "collector_timeout_seconds": 0.05,
                "collection_timeout_seconds": 1,
            }
        )
        report = await collect_support_bundle(
            _context(app=app, limits=limits),
            (
                FunctionalSupportBundleCollector("before", succeed),
                operational,
                FunctionalSupportBundleCollector("after", succeed),
            ),
        )
        return report, checkpoint_started.is_set(), cancellation_counts

    report, checkpoint_started, cancellation_counts = asyncio.run(scenario())

    assert checkpoint_started is True
    assert cancellation_counts and cancellation_counts[0] >= 1
    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [(item.name, item.disposition, item.reason_code) for item in report.collectors] == [
        ("before", CollectorDisposition.COLLECTED, None),
        (collector_name, CollectorDisposition.TIMED_OUT, "collector_deadline_elapsed"),
        ("after", CollectorDisposition.COLLECTED, None),
    ]


def test_runner_rejects_a_late_result_after_timeout_cancellation_is_suppressed() -> None:
    async def suppress_timeout(_context) -> SupportCollectorOutput:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return collected(_runtime_evidence("late"))
        raise AssertionError("unset event unexpectedly completed")

    async def succeed(_context) -> SupportCollectorOutput:
        return collected(_runtime_evidence("later"))

    limits = _context().limits.model_copy(
        update={
            "collector_timeout_seconds": 0.01,
            "collection_timeout_seconds": 1,
        }
    )
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (
                FunctionalSupportBundleCollector("late", suppress_timeout),
                FunctionalSupportBundleCollector("later", succeed),
            ),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [item.disposition for item in report.collectors] == [
        CollectorDisposition.TIMED_OUT,
        CollectorDisposition.COLLECTED,
    ]
    assert report.collectors[0].reason_code == "collector_deadline_elapsed"


def test_runner_applies_collector_deadline_through_evidence_validation(monkeypatch) -> None:
    app = CayuApp(enable_logging=False)
    original_redact_json = app.redact_json
    redaction_calls = 0
    block_redaction = False
    redaction_started = threading.Event()
    release_redaction = threading.Event()

    def delay_first_redaction(value):
        nonlocal redaction_calls
        redaction_calls += 1
        if block_redaction and redaction_calls == 1:
            redaction_started.set()
            if not release_redaction.wait(timeout=5):
                raise TimeoutError("redaction test barrier was not released")
        return original_redact_json(value)

    monkeypatch.setattr(app, "redact_json", delay_first_redaction)

    async def succeed(_context) -> SupportCollectorOutput:
        return collected(_runtime_evidence())

    limits = _context(app=app).limits.model_copy(
        update={
            "collector_timeout_seconds": 0.01,
            "collection_timeout_seconds": 1,
        }
    )
    context = _context(app=app, limits=limits)
    redaction_calls = 0
    block_redaction = True
    started = time.monotonic()
    try:
        report = asyncio.run(
            collect_support_bundle(
                context,
                (
                    FunctionalSupportBundleCollector("slow_validation", succeed),
                    FunctionalSupportBundleCollector("later", succeed),
                ),
            )
        )
    finally:
        release_redaction.set()

    assert time.monotonic() - started < 0.5
    assert redaction_started.is_set()
    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [item.disposition for item in report.collectors] == [
        CollectorDisposition.TIMED_OUT,
        CollectorDisposition.COLLECTED,
    ]
    assert report.collectors[0].reason_code == "collector_deadline_elapsed"


def test_optional_package_inventory_cooperates_with_collector_deadline(monkeypatch) -> None:
    calls = 0
    lookup_started = threading.Event()
    release_lookup = threading.Event()

    def slow_version(distribution: str) -> str:
        nonlocal calls
        calls += 1
        lookup_started.set()
        if not release_lookup.wait(timeout=5):
            raise TimeoutError("package lookup test barrier was not released")
        raise support_bundles.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(support_bundles.metadata, "version", slow_version)
    limits = _context().limits.model_copy(
        update={
            "collector_timeout_seconds": 0.05,
            "collection_timeout_seconds": 1,
        }
    )
    collectors = tuple(
        collector
        for collector in builtin_support_collectors()
        if collector.name in {"optional_packages", "artifacts"}
    )

    started = time.monotonic()
    try:
        report = asyncio.run(collect_support_bundle(_context(limits=limits), collectors))
        elapsed = time.monotonic() - started
    finally:
        release_lookup.set()
    results = {item.name: item for item in report.collectors}

    assert elapsed < 0.5
    assert lookup_started.is_set()
    assert calls == 1
    assert results["optional_packages"].disposition is CollectorDisposition.TIMED_OUT
    assert results["optional_packages"].reason_code == "collector_deadline_elapsed"
    assert results["artifacts"].disposition is CollectorDisposition.COLLECTED


def test_sync_evidence_preparation_preserves_owner_task_cancellation(monkeypatch) -> None:
    app = CayuApp(enable_logging=False)
    original_redact_json = app.redact_json
    block_redaction = False
    redaction_started = threading.Event()
    release_redaction = threading.Event()

    def blocked_redaction(value):
        if block_redaction:
            redaction_started.set()
            if not release_redaction.wait(timeout=5):
                raise TimeoutError("redaction cancellation barrier was not released")
        return original_redact_json(value)

    monkeypatch.setattr(app, "redact_json", blocked_redaction)
    context = _context(app=app)
    block_redaction = True

    async def succeed(_context) -> SupportCollectorOutput:
        return collected(_runtime_evidence())

    async def run() -> None:
        owner = asyncio.create_task(
            collect_support_bundle(
                context,
                (FunctionalSupportBundleCollector("cancelled", succeed),),
            )
        )
        while not redaction_started.is_set():
            await asyncio.sleep(0)
        owner.cancel()
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled() is True

    try:
        asyncio.run(run())
    finally:
        release_redaction.set()


def test_sync_evidence_preparation_types_child_cancellation_and_continues(
    monkeypatch,
) -> None:
    app = CayuApp(enable_logging=False)
    original_redact_json = app.redact_json
    context = _context(app=app)
    calls = 0

    def child_cancel_first_redaction(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        return original_redact_json(value)

    monkeypatch.setattr(app, "redact_json", child_cancel_first_redaction)

    async def succeed(_context) -> SupportCollectorOutput:
        return collected(_runtime_evidence())

    report = asyncio.run(
        collect_support_bundle(
            context,
            (
                FunctionalSupportBundleCollector("child_cancel", succeed),
                FunctionalSupportBundleCollector("later", succeed),
            ),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [(item.disposition, item.reason_code) for item in report.collectors] == [
        (
            CollectorDisposition.FAILED,
            "collector_cancelled_without_task_cancellation",
        ),
        (CollectorDisposition.COLLECTED, None),
    ]


def test_builtin_sqlite_operational_collectors_interrupt_slow_reads(
    tmp_path: Path,
) -> None:
    session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    slow_queries = True

    def slow_progress() -> int:
        if slow_queries:
            time.sleep(0.01)
        return 0

    session_store._read_connection.set_progress_handler(slow_progress, 1)
    task_store._connection.set_progress_handler(slow_progress, 1)
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    limits = _context(app=app).limits.model_copy(
        update={
            "collector_timeout_seconds": 0.05,
            "collection_timeout_seconds": 1,
        }
    )
    collectors = tuple(
        collector
        for collector in builtin_support_collectors()
        if collector.name in {"sessions", "tasks", "artifacts"}
    )

    async def exercise() -> tuple[SupportBundleReport, float, int, bool]:
        nonlocal slow_queries

        started = time.monotonic()
        try:
            report = await collect_support_bundle(
                _context(app=app, limits=limits),
                collectors,
            )
        finally:
            slow_queries = False
            await session_store.close()
            await task_store.close()
            await asyncio.sleep(0)
        return (
            report,
            time.monotonic() - started,
            len(session_store._detached_read_tasks),
            task_store._lock.locked(),
        )

    report, elapsed, detached_reads, task_locked = asyncio.run(exercise())
    results = {item.name: item for item in report.collectors}

    assert elapsed < 1
    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert results["sessions"].disposition is CollectorDisposition.TIMED_OUT
    assert results["tasks"].disposition is CollectorDisposition.TIMED_OUT
    assert results["artifacts"].disposition is CollectorDisposition.COLLECTED
    assert detached_reads == 0
    assert task_locked is False


def test_runner_discards_evidence_changed_by_application_redaction(monkeypatch) -> None:
    app = CayuApp(enable_logging=False)
    context = _context(app=app)
    monkeypatch.setattr(app, "redact_json", lambda _value: {"redacted": True})

    async def expose(_context):
        return collected(_runtime_evidence("secret-canary"))

    report = asyncio.run(
        collect_support_bundle(
            context,
            (FunctionalSupportBundleCollector("unsafe", expose),),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert len(report.collectors) == 1
    assert report.collectors[0].name == "unsafe"
    assert report.collectors[0].disposition is CollectorDisposition.REDACTED
    assert report.collectors[0].evidence_bytes == 0
    assert report.collectors[0].reason_code == ("application_redaction_changed_evidence")
    assert "secret-canary" not in report.model_dump_json()


def test_runner_owns_the_exact_evidence_that_passed_redaction(
    monkeypatch,
    caplog,
    capsys,
    recwarn,
) -> None:
    safe_value = "safe-value-000000"
    secret_value = "secret-token-0000"
    assert len(safe_value) == len(secret_value)
    app = CayuApp(enable_logging=False)
    monkeypatch.setattr(app, "redact_json", SecretRedactor(secret_value).redact_json)
    evidence = _runtime_evidence(safe_value)
    retained_output = collected(evidence)

    async def first(_context) -> SupportCollectorOutput:
        return retained_output

    async def mutate_after_acceptance(_context) -> SupportCollectorOutput:
        object.__setattr__(evidence, "cayu_version", secret_value)
        return collected(_runtime_evidence("later"))

    report = asyncio.run(
        collect_support_bundle(
            _context(app=app),
            (
                FunctionalSupportBundleCollector("first", first),
                FunctionalSupportBundleCollector("mutator", mutate_after_acceptance),
            ),
        )
    )
    payload = encode_support_bundle(report)
    validated = validate_support_bundle_archive(payload)
    output = capsys.readouterr()

    assert report.outcome is SupportBundleOutcome.CLEAN
    assert isinstance(report.collectors[0].evidence, RuntimeIdentityEvidence)
    assert report.collectors[0].evidence.cayu_version == safe_value
    assert validated == report
    assert secret_value not in payload.decode("latin-1")
    assert secret_value not in output.out + output.err + caplog.text
    assert all(secret_value not in str(item.message) for item in recwarn)


def test_runner_isolates_forbidden_collector_evidence_before_report_publication() -> None:
    async def unsafe(_context):
        return collected(_runtime_evidence("/private/collector-path-canary"))

    async def safe(_context):
        return collected(_runtime_evidence("safe-version"))

    report = asyncio.run(
        collect_support_bundle(
            _context(),
            (
                FunctionalSupportBundleCollector("unsafe", unsafe),
                FunctionalSupportBundleCollector("safe", safe),
            ),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert report.collectors[0].disposition is CollectorDisposition.REDACTED
    assert report.collectors[0].reason_code == "collector_evidence_forbidden_content"
    assert report.collectors[0].evidence is None
    assert report.collectors[1].disposition is CollectorDisposition.COLLECTED
    payload = encode_support_bundle(report)
    assert validate_support_bundle_archive(payload) == report
    assert b"collector-path-canary" not in payload


@pytest.mark.parametrize(
    "unsafe_value",
    ("/private/diagnostic/path", "failure at /private/diagnostic/path", r"at C:\private\x"),
)
def test_archive_validation_rejects_absolute_paths(unsafe_value: str) -> None:
    context = _context()
    evidence = _runtime_evidence(unsafe_value)
    evidence_bytes = len(
        json.dumps(
            evidence.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    report = SupportBundleReport.from_results(
        generated_at=datetime(2026, 1, 2, tzinfo=UTC),
        outcome=SupportBundleOutcome.CLEAN,
        limits=context.limits,
        collection_duration_ms=0,
        collectors=(
            SupportCollectorResult(
                name="runtime",
                disposition=CollectorDisposition.COLLECTED,
                duration_ms=0,
                evidence_bytes=evidence_bytes,
                evidence=evidence,
            ),
        ),
    )

    with pytest.raises(ValueError, match="absolute path"):
        encode_support_bundle(report)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission and symlink contract")
def test_atomic_writer_replaces_regular_file_with_mode_0600_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    destination = tmp_path / "support.zip"
    destination.write_text("old", encoding="utf-8")
    os.chmod(destination, 0o644)

    write_support_bundle_atomic(destination, payload)

    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []

    destination.unlink()
    outside = tmp_path / "outside"
    outside.write_text("unchanged", encoding="utf-8")
    destination.symlink_to(outside)
    with pytest.raises(OSError, match="regular file"):
        write_support_bundle_atomic(destination, payload)
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert destination.is_symlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX stable parent alias contract")
def test_atomic_writer_accepts_stable_symlink_parent_alias(tmp_path: Path) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    canonical = tmp_path / "canonical"
    nested = canonical / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(canonical, target_is_directory=True)

    write_support_bundle_atomic(alias / "nested" / "support.zip", payload)

    assert (nested / "support.zip").read_bytes() == payload
    assert stat.S_IMODE((nested / "support.zip").stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX staging identity contract")
def test_atomic_writer_rejects_staging_name_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    destination = tmp_path / "support.zip"
    original_replace = os.replace
    substituted = False

    def substitute_staging_entry(
        source,
        target,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal substituted
        substituted = True
        displaced = ".displaced-support-bundle"
        os.rename(source, displaced, src_dir_fd=src_dir_fd, dst_dir_fd=src_dir_fd)
        replacement_fd = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        try:
            replacement = b"x" * len(payload)
            offset = 0
            while offset < len(replacement):
                offset += os.write(replacement_fd, replacement[offset:])
        finally:
            os.close(replacement_fd)
        os.unlink(displaced, dir_fd=src_dir_fd)
        original_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", substitute_staging_entry)

    with pytest.raises(OSError, match="identity"):
        write_support_bundle_atomic(destination, payload)

    assert substituted is True
    assert destination.read_bytes() != payload
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX staging content contract")
def test_atomic_writer_rejects_in_place_staging_content_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    destination = tmp_path / "support.zip"
    original_replace = os.replace

    def corrupt_staging_content(
        source,
        target,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        replacement_fd = os.open(source, os.O_WRONLY, dir_fd=src_dir_fd)
        try:
            replacement = b"x" * len(payload)
            offset = 0
            while offset < len(replacement):
                offset += os.write(replacement_fd, replacement[offset:])
            os.fsync(replacement_fd)
        finally:
            os.close(replacement_fd)
        original_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", corrupt_staging_content)

    with pytest.raises(OSError, match="content changed"):
        write_support_bundle_atomic(destination, payload)

    assert destination.read_bytes() != payload
    assert list(tmp_path.glob(".support.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent traversal contract")
def test_atomic_writer_does_not_follow_parent_replaced_by_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    safe = tmp_path / "safe"
    safe_nested = safe / "nested"
    safe_nested.mkdir(parents=True)
    moved = tmp_path / "moved"
    redirect = tmp_path / "redirect"
    redirect_nested = redirect / "nested"
    redirect_nested.mkdir(parents=True)
    destination = safe_nested / "support.zip"
    original_open = os.open
    swapped = False

    def swap_before_component_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "safe" and dir_fd is not None and not swapped:
            swapped = True
            safe.rename(moved)
            safe.symlink_to(redirect, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_component_open)

    with pytest.raises(OSError):
        write_support_bundle_atomic(destination, payload)

    assert swapped is True
    assert not (redirect_nested / "support.zip").exists()
    assert not (moved / "nested" / "support.zip").exists()
    assert list(redirect_nested.glob(".support.zip.cayu-doctor-*.tmp")) == []
    assert list((moved / "nested").glob(".support.zip.cayu-doctor-*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent identity contract")
def test_atomic_writer_detects_parent_replaced_after_directory_is_opened(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="test_boot_failure",
        )
    )
    safe = tmp_path / "safe"
    safe_nested = safe / "nested"
    safe_nested.mkdir(parents=True)
    moved = tmp_path / "moved"
    redirect = tmp_path / "redirect"
    redirect_nested = redirect / "nested"
    redirect_nested.mkdir(parents=True)
    destination = safe_nested / "support.zip"
    original_open = os.open
    swapped = False

    def swap_after_parent_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            safe.rename(moved)
            safe.symlink_to(redirect, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(os, "open", swap_after_parent_open)

    with pytest.raises(OSError):
        write_support_bundle_atomic(destination, payload)

    published = moved / "nested" / "support.zip"
    assert swapped is True
    assert published.read_bytes() == payload
    assert stat.S_IMODE(published.stat().st_mode) == 0o600
    assert not (redirect_nested / "support.zip").exists()
    assert list(redirect_nested.glob(".support.zip.cayu-doctor-*.tmp")) == []
    assert list((moved / "nested").glob(".support.zip.cayu-doctor-*.tmp")) == []


def test_explicit_event_tail_is_envelope_only_and_bounded() -> None:

    session_id = "private-session-canary"

    async def collect_tail() -> SupportBundleReport:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                environment_name="local",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        for number in range(3):
            await store.append_event(
                session_id,
                Event(
                    type=f"custom.secret-type-{number}",
                    session_id=session_id,
                    payload={
                        "transcript": "model-text-canary",
                        "tool_arguments": {"token": "tool-secret-canary"},
                    },
                ),
            )
        app = CayuApp(session_store=store, enable_logging=False)
        limits = _context(app=app).limits.model_copy(update={"event_limit": 2})
        return await collect_support_bundle(
            _context(app=app, limits=limits),
            builtin_support_collectors(session_selectors=(session_id,)),
        )

    report = asyncio.run(collect_tail())

    tail_result = next(item for item in report.collectors if item.name == "session_events.1")
    assert tail_result.disposition is CollectorDisposition.COLLECTED
    assert isinstance(tail_result.evidence, SessionEventTailEvidence)
    assert tail_result.evidence.projection == "redacted_envelope_only"
    assert tail_result.evidence.returned_count == 2
    assert tail_result.evidence.omitted_count_lower_bound == 1
    assert tail_result.evidence.omitted_count_exact is False
    assert tail_result.evidence.tail_complete is False
    assert tail_result.evidence.first_sequence == 2
    assert tail_result.evidence.last_sequence == 3
    assert tail_result.evidence.first_timestamp == tail_result.evidence.events[0].timestamp
    assert tail_result.evidence.last_timestamp == tail_result.evidence.events[-1].timestamp
    assert [item.sequence for item in tail_result.evidence.events] == [2, 3]
    assert {item.type for item in tail_result.evidence.events} == {"custom.redacted"}

    serialized = report.model_dump_json()
    assert session_id not in serialized
    assert "model-text-canary" not in serialized
    assert "tool-secret-canary" not in serialized
    assert "secret-type" not in serialized


def test_event_tail_maximum_keeps_one_record_for_completeness_proof() -> None:
    observed_query_limits: list[int] = []

    class InspectingSessionStore(InMemorySessionStore):
        async def query_events_bounded(self, query, *, max_bytes):
            observed_query_limits.append(query.limit)
            return await super().query_events_bounded(query, max_bytes=max_bytes)

    async def scenario() -> SupportBundleReport:
        store = InspectingSessionStore()
        session_id = "maximum-tail-session"
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                environment_name="local",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        values = _context(app=app).limits.model_dump()
        values["event_limit"] = 4999
        limits = SupportBundleLimits.model_validate(values)
        return await collect_support_bundle(
            _context(app=app, limits=limits),
            builtin_support_collectors(session_selectors=(session_id,)),
        )

    report = asyncio.run(scenario())

    tail_result = next(item for item in report.collectors if item.name == "session_events.1")
    assert tail_result.disposition is CollectorDisposition.COLLECTED
    assert observed_query_limits == [5000]
    invalid_values = _context().limits.model_dump()
    invalid_values["event_limit"] = 5000
    with pytest.raises(ValueError):
        SupportBundleLimits.model_validate(invalid_values)


def test_command_limit_reserves_time_beyond_worker_and_publication() -> None:
    values = _context().limits.model_dump()
    values.update(
        worker_timeout_seconds=10,
        publication_timeout_seconds=10,
        reconciliation_timeout_seconds=5,
        command_timeout_seconds=20,
    )

    with pytest.raises(ValueError, match="leave command teardown time"):
        SupportBundleLimits.model_validate(values)


def test_no_event_collector_is_registered_without_explicit_session() -> None:
    assert all(
        not collector.name.startswith("session_events.")
        for collector in builtin_support_collectors()
    )


def test_builtin_collectors_only_register_authoritative_health_evidence() -> None:
    names = {collector.name for collector in builtin_support_collectors()}

    assert "recovery_cleanup" in names
    assert names.isdisjoint(
        {
            "workers_and_leases",
            "event_side_effect_health",
            "handoff_and_recovery",
            "environment_health",
            "provider_live_health",
        }
    )


def test_artifact_collector_omits_default_local_store_identity_and_digest(
    tmp_path: Path,
) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "private-artifacts")
    artifact_store_id = artifact_store.id
    artifact_store_fingerprint = f"sha256:{sha256(artifact_store_id.encode()).hexdigest()}"
    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="artifact-environment"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    collector = next(item for item in builtin_support_collectors() if item.name == "artifacts")

    report = asyncio.run(collect_support_bundle(_context(app=app), (collector,)))
    payload = encode_support_bundle(report)
    validated = validate_support_bundle_archive(payload)

    evidence = report.collectors[0].evidence
    assert isinstance(evidence, ArtifactAvailabilityEvidence)
    assert evidence.model_dump(mode="json") == {
        "kind": "artifact_availability",
        "registered": True,
        "registration_count": 1,
        "availability": "configured_only_not_live_verified",
    }
    unsafe_evidence = evidence.model_dump(mode="json")
    unsafe_evidence["registration_fingerprints"] = [artifact_store_fingerprint]
    with pytest.raises(ValueError):
        ArtifactAvailabilityEvidence.model_validate(unsafe_evidence)
    with pytest.raises(ValueError, match="must match its count"):
        ArtifactAvailabilityEvidence(registered=False, registration_count=1)
    assert validated == report
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        report_payload = archive.read("report.json")
        summary_payload = archive.read("summary.txt")
    for private_value in (artifact_store_id, artifact_store_fingerprint):
        encoded = private_value.encode()
        assert encoded not in report_payload
        assert encoded not in summary_payload
        assert encoded not in payload


def test_recovery_cleanup_collector_bounds_and_reconstructs_typed_status(
    monkeypatch,
) -> None:
    policy = RecoveryCleanupPolicy(
        step_timeout_seconds=1.5,
        overall_timeout_seconds=4,
        max_supervised_tasks=7,
    )
    app = CayuApp(recovery_cleanup_policy=policy, enable_logging=False)
    snapshot = RecoveryCleanupSupervisorSnapshot(
        active_tasks=2,
        retained_tasks=3,
        timed_out_steps=5,
        completed_after_timeout=1,
        failed_after_timeout=2,
        retained_after_cancellation=1,
        capacity_exhausted_steps=4,
        retained=tuple(
            RecoveryCleanupRetainedTaskSnapshot(
                operation=f"recovery operation {index}",
                scope=(
                    RecoveryCleanupDeadlineScope.STEP
                    if index % 2
                    else RecoveryCleanupDeadlineScope.OVERALL
                ),
                timeout_seconds=float(index),
                caller_cancellation_observed=index == 3,
            )
            for index in range(1, 4)
        ),
    )
    monkeypatch.setattr(app, "recovery_cleanup_status", lambda: snapshot)
    limits = _context(app=app).limits.model_copy(update={"max_items": 2})
    collector = next(
        item for item in builtin_support_collectors() if item.name == "recovery_cleanup"
    )

    report = asyncio.run(
        collect_support_bundle(
            _context(app=app, limits=limits),
            (collector,),
        )
    )
    validated = validate_support_bundle_archive(encode_support_bundle(report))

    assert report.outcome is SupportBundleOutcome.CLEAN
    evidence = report.collectors[0].evidence
    assert isinstance(evidence, RecoveryCleanupEvidence)
    assert evidence.policy.model_dump(mode="json") == {
        "step_timeout_seconds": 1.5,
        "overall_timeout_seconds": 4.0,
        "max_supervised_tasks": 7,
    }
    assert evidence.snapshot.active_tasks == 2
    assert evidence.snapshot.retained_tasks == 3
    assert evidence.snapshot.retained_inventory.model_dump(mode="json") == {
        "total_count": 3,
        "included_count": 2,
        "truncated": True,
    }
    assert [item.operation for item in evidence.snapshot.retained] == [
        "recovery operation 1",
        "recovery operation 2",
    ]
    assert validated == report


def test_recovery_cleanup_evidence_rejects_untyped_or_unbounded_values() -> None:
    with pytest.raises(ValueError):
        RecoveryCleanupPolicyEvidence.model_validate(
            {
                "step_timeout_seconds": True,
                "overall_timeout_seconds": 4.0,
                "max_supervised_tasks": 7,
            }
        )
    with pytest.raises(ValueError):
        RecoveryCleanupPolicyEvidence.model_validate(
            {
                "step_timeout_seconds": 1.0,
                "overall_timeout_seconds": 86_401.0,
                "max_supervised_tasks": 7,
            }
        )
    with pytest.raises(ValueError):
        RecoveryCleanupRetainedEvidence.model_validate(
            {
                "operation": "recovery operation",
                "scope": "step",
                "timeout_seconds": 1.0,
                "caller_cancellation_observed": 1,
            }
        )


def test_builtin_collectors_report_configured_task_and_mcp_without_server_names() -> None:
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        mcp_manifest_policy=McpManifestPolicy(),
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="diagnostic-environment"),
            mcp_servers=(
                McpServerSpec(
                    name="mcp-private-name-canary",
                    command=["mcp-server"],
                ),
            ),
        ),
        default=True,
    )

    report = asyncio.run(
        collect_support_bundle(
            _context(app=app),
            builtin_support_collectors(),
        )
    )
    results = {item.name: item for item in report.collectors}

    manifest = results["manifest"].evidence
    assert isinstance(manifest, ManifestSummaryEvidence)
    assert manifest.mcp_manifest_policy_configured is True
    assert manifest.environments[0].mcp_server_count == 1
    assert manifest.environments[0].workspace_branch_capabilities.model_dump(mode="json") == {
        "isolation": False,
        "net_changes": False,
        "publication": "unsupported",
        "recovery": "unsupported",
        "retention": "unsupported",
        "lifecycle_inspection": "unsupported",
        "detail_code": "workspace_branching_unsupported",
    }
    assert manifest.environments[0].workspace_branch_lifecycle.attached_count == 0
    assert results["control_plane"].reason_code == "maintained_service_not_selected"
    assert "mcp-private-name-canary" not in report.model_dump_json()
    recovery_cleanup = results["recovery_cleanup"].evidence
    assert isinstance(recovery_cleanup, RecoveryCleanupEvidence)
    assert recovery_cleanup.snapshot.active_tasks == 0
    assert recovery_cleanup.snapshot.retained_tasks == 0
    assert recovery_cleanup.snapshot.retained_inventory.total_count == 0
    assert isinstance(results["tasks"].evidence, TaskOperationalEvidence)
    assert results["tasks"].disposition is CollectorDisposition.COLLECTED


def test_store_readiness_fails_closed_without_rendering_mutated_builtin_path() -> None:
    class UnsafePath:
        def __str__(self) -> str:
            raise AssertionError("mutated-store-path-secret-canary")

    async def scenario() -> SupportBundleReport:
        store = SQLiteSessionStore(":memory:")
        app = CayuApp(session_store=store, enable_logging=False)
        context = _context(app=app)
        object.__setattr__(store, "path", UnsafePath())
        stores_collector = next(
            item for item in builtin_support_collectors() if item.name == "stores"
        )
        try:
            return await collect_support_bundle(context, (stores_collector,))
        finally:
            await store.close()

    report = asyncio.run(scenario())

    result = report.collectors[0]
    assert result.disposition is CollectorDisposition.COLLECTED
    assert isinstance(result.evidence, StoreSummaryEvidence)
    descriptor = next(item for item in result.evidence.stores if item.role == "session")
    assert descriptor.schema_readiness == "unavailable"
    assert "mutated-store-path-secret-canary" not in report.model_dump_json()


def test_sqlite_event_tail_round_trips_without_private_payload(
    tmp_path: Path,
) -> None:
    session_id = "sqlite-private-session-canary"

    async def collect_tail() -> SupportBundleReport:
        store = SQLiteSessionStore(tmp_path / "support.sqlite")
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    environment_name="local",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_event(
                session_id,
                Event(
                    type="custom.sqlite-secret-type",
                    session_id=session_id,
                    payload={"transcript": "sqlite-model-text-canary"},
                ),
            )
            app = CayuApp(session_store=store, enable_logging=False)
            return await collect_support_bundle(
                _context(app=app),
                builtin_support_collectors(session_selectors=(session_id,)),
            )
        finally:
            await store.close()

    report = asyncio.run(collect_tail())
    validated = validate_support_bundle_archive(encode_support_bundle(report))

    tail_result = next(item for item in validated.collectors if item.name == "session_events.1")
    assert tail_result.disposition is CollectorDisposition.COLLECTED
    serialized = validated.model_dump_json()
    assert session_id not in serialized
    assert "sqlite-model-text-canary" not in serialized
    assert "sqlite-secret-type" not in serialized


def test_each_collection_has_a_fresh_bundle_identity() -> None:
    first = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="first_boot_failure",
    )
    second = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="second_boot_failure",
    )

    assert first.bundle_id != second.bundle_id


def test_runner_rejects_malformed_and_oversized_collector_results() -> None:
    async def malformed(_context):
        return object()

    async def oversized(_context):
        return collected(_runtime_evidence("x" * 2048))

    limits = _context().limits.model_copy(update={"max_collector_bytes": 1024})
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (
                FunctionalSupportBundleCollector("malformed", malformed),
                FunctionalSupportBundleCollector("oversized", oversized),
            ),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [
        (item.name, item.disposition, item.reason_code, item.evidence_bytes)
        for item in report.collectors
    ] == [
        (
            "malformed",
            CollectorDisposition.FAILED,
            "invalid_collector_output",
            0,
        ),
        (
            "oversized",
            CollectorDisposition.FAILED,
            "collector_result_too_large",
            0,
        ),
    ]


def test_collection_byte_limit_preserves_success_and_skips_remaining_collectors() -> None:
    executed: list[str] = []

    async def first(_context):
        executed.append("first")
        return collected(_runtime_evidence("a" * 600))

    async def second(_context):
        executed.append("second")
        return collected(_runtime_evidence("b" * 600))

    async def must_not_run(_context):
        executed.append("third")
        raise AssertionError("collector after byte exhaustion must not run")

    limits = _context().limits.model_copy(
        update={
            "max_collector_bytes": 2048,
            "max_evidence_bytes": 1024,
        }
    )
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (
                FunctionalSupportBundleCollector("first", first),
                FunctionalSupportBundleCollector("second", second),
                FunctionalSupportBundleCollector("third", must_not_run),
            ),
        )
    )

    assert executed == ["first", "second"]
    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [item.disposition for item in report.collectors] == [
        CollectorDisposition.COLLECTED,
        CollectorDisposition.SKIPPED,
        CollectorDisposition.SKIPPED,
    ]
    assert [item.reason_code for item in report.collectors[1:]] == [
        "bundle_evidence_byte_limit_reached",
        "bundle_evidence_byte_limit_reached",
    ]
    assert report.total_evidence_bytes == report.collectors[0].evidence_bytes
    assert report.collected_count == 1
    assert report.omitted_count == 2
    assert report.evidence_complete is False


def test_event_tail_authority_and_store_failures_are_typed_and_secret_safe() -> None:
    valid_session = "valid-private-session-canary"
    unauthorized = "cayu_authority_invalid-secret-canary"
    missing = "missing-private-session-canary"

    class FailingEventStore(InMemorySessionStore):
        async def query_events_bounded(self, query, *, max_bytes):
            del query, max_bytes
            raise RuntimeError("store-failure-secret-canary")

    async def scenario() -> SupportBundleReport:
        store = FailingEventStore()
        await store.create(
            RunRequest(
                session_id=valid_session,
                agent_name="assistant",
                environment_name="local",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        return await collect_support_bundle(
            _context(app=app),
            builtin_support_collectors(session_selectors=(valid_session, unauthorized, missing)),
        )

    report = asyncio.run(scenario())
    results = {item.name: item for item in report.collectors}

    assert results["session_events.1"].disposition is CollectorDisposition.FAILED
    assert results["session_events.1"].reason_code == "collector_failed"
    assert results["session_events.2"].disposition is CollectorDisposition.FAILED
    assert results["session_events.2"].reason_code == "collector_failed"
    assert results["session_events.3"].disposition is CollectorDisposition.UNAVAILABLE
    assert results["session_events.3"].reason_code == "session_not_found"
    serialized = report.model_dump_json()
    assert "store-failure-secret-canary" not in serialized
    assert valid_session not in serialized
    assert unauthorized not in serialized
    assert missing not in serialized


def test_archive_validation_rejects_non_restrictive_member_permissions() -> None:
    payload = encode_support_bundle(
        minimal_support_bundle_report(
            outcome=SupportBundleOutcome.BOOT_FAILED,
            reason_code="permission_test",
        )
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        members = {name: source.read(name) for name in ("report.json", "summary.txt")}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, member_payload in members.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, member_payload)

    with pytest.raises(ValueError, match="unsafe"):
        validate_support_bundle_archive(stream.getvalue())


def test_external_cancellation_propagates_with_asyncio_task_state() -> None:
    async def scenario() -> tuple[int, bool]:
        started = asyncio.Event()

        async def wait_forever(_context) -> SupportCollectorOutput:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unset event unexpectedly completed")

        task = asyncio.create_task(
            collect_support_bundle(
                _context(),
                (FunctionalSupportBundleCollector("waiting", wait_forever),),
            )
        )
        await started.wait()
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task.cancelling(), task.cancelled()

    cancelling_count, cancelled = asyncio.run(scenario())

    assert cancelling_count >= 2
    assert cancelled is True


def test_child_created_cancelled_error_is_a_typed_collector_failure() -> None:
    async def child_cancel(_context) -> SupportCollectorOutput:
        raise asyncio.CancelledError

    report = asyncio.run(
        collect_support_bundle(
            _context(),
            (FunctionalSupportBundleCollector("child_cancel", child_cancel),),
        )
    )

    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert report.collectors[0].disposition is CollectorDisposition.FAILED
    assert report.collectors[0].reason_code == ("collector_cancelled_without_task_cancellation")


def test_collection_deadline_skips_collectors_that_never_started() -> None:
    executed: list[str] = []

    async def consume_deadline(_context) -> SupportCollectorOutput:
        executed.append("first")
        await asyncio.Event().wait()
        raise AssertionError("unset event unexpectedly completed")

    async def must_not_run(_context) -> SupportCollectorOutput:
        executed.append("second")
        raise AssertionError("collector after collection deadline must not run")

    limits = _context().limits.model_copy(
        update={
            "collector_timeout_seconds": 1,
            "collection_timeout_seconds": 0.01,
        }
    )
    report = asyncio.run(
        collect_support_bundle(
            _context(limits=limits),
            (
                FunctionalSupportBundleCollector("first", consume_deadline),
                FunctionalSupportBundleCollector("second", must_not_run),
            ),
        )
    )

    assert executed == ["first"]
    assert report.outcome is SupportBundleOutcome.PARTIAL
    assert [item.disposition for item in report.collectors] == [
        CollectorDisposition.TIMED_OUT,
        CollectorDisposition.SKIPPED,
    ]
    assert [item.reason_code for item in report.collectors] == [
        "collector_deadline_elapsed",
        "collection_deadline_elapsed",
    ]
    assert report.collectors[1].duration_ms == 0
    assert report.evidence_complete is False


def test_archive_validation_rejects_duplicate_key_hiding_forbidden_text() -> None:
    report = minimal_support_bundle_report(
        outcome=SupportBundleOutcome.BOOT_FAILED,
        reason_code="canonical_report_test",
    )
    payload = encode_support_bundle(report)
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        report_text = source.read("report.json").decode("utf-8")
        summary = source.read("summary.txt")
    bundle_id_line = f'  "bundle_id": "{report.bundle_id}",'
    tampered_report = report_text.replace(
        bundle_id_line,
        '  "bundle_id": "postgresql://archive-secret-canary",\n' + bundle_id_line,
        1,
    ).encode("utf-8")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, member_payload in (
            ("report.json", tampered_report),
            ("summary.txt", summary),
        ):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(member, member_payload)

    with pytest.raises(ValueError, match="canonical"):
        validate_support_bundle_archive(stream.getvalue())
