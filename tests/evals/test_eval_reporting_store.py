from __future__ import annotations

import asyncio
import copy

import pytest
from tests.server.test_server_eval_reporting import _payload


async def _report_conformance(store):
    from cayu.evals.read_views import EvalViewQuery
    from cayu.evals.reported import EvalReport, EvalReportConflict

    payload = await _payload("ci-target")
    report = EvalReport.model_validate(payload)

    async def publish(value):
        return await store.publish_report(
            value, publisher_id="stable-publisher", redact_json=lambda value: value
        )

    original, repeated = await asyncio.gather(publish(report), publish(report))
    assert original == repeated
    assert await store.load_report(original.receipt.id) == original
    for path, value in (
        (("target_key",), "different-target"),
        (("program_key",), "different-program"),
        (("declared_suite_revision",), "sha256:" + "1" * 64),
        (("origin", "branch"), "different-branch"),
        (("origin", "commit"), "c" * 40),
        (("evidence", "run", "suite_id"), "different-suite"),
        (("evidence", "run", "duration_ms"), report.evidence.run.duration_ms + 1),
    ):
        changed = copy.deepcopy(payload)
        current = changed
        for part in path[:-1]:
            current = current[part]
        current[path[-1]] = value
        with pytest.raises(EvalReportConflict):
            await publish(EvalReport.model_validate(changed))
        assert await store.load_report(original.receipt.id) == original
    catalog = await store.list_view_index(EvalViewQuery(target_key="ci-target"))
    assert len(catalog.items) == 1 and catalog.items[0].source == "reported_suite"
    history = await store.list_view_index(EvalViewQuery(suite=original.series.selector))
    assert len(history.items) == 1 and history.items[0].revision == original.receipt.id
    assert await publish(report) == original
    # An association rejection must roll back the whole operation on every
    # backend, so the same exact operation succeeds once its corpus exists.
    from tests.evals.test_corpus_execution import _corpus, _provider, _target

    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.reported import EvalCorpusReportEvidence, EvalReportCorpusAssociation

    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
    associated = EvalReport(
        report_id="corpus-operation",
        target_key=target.key,
        program_key="corpus-program",
        evidence=EvalCorpusReportEvidence(result=result),
        association=EvalReportCorpusAssociation(
            corpus_revision=corpus.revision, suite_id=corpus.suites[0].id
        ),
    )
    with pytest.raises(EvalReportConflict):
        await publish(associated)
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    accepted = await publish(associated)
    assert await publish(associated) == accepted
    assert await store.load_result_record(result.revision) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_report_store_exact_tuple_concurrent_replay(tmp_path, backend):
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    async def exercise():
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "reports.db")
        )
        try:
            await _report_conformance(store)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_report_store_exact_tuple_concurrent_replay(postgres_dsn):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await _report_conformance(store)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_report_store_rejects_mutated_secret_objects_without_diagnostics(
    tmp_path, backend, capsys, caplog
):
    import warnings

    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    async def exercise():
        from cayu.evals.reported import EvalReport

        payload = await _payload("ci-target")
        report = EvalReport.model_validate(payload)

        class Hostile:
            def __repr__(self):
                return "MUTATED-REPORT-PRIVATE-CANARY"

            __str__ = __repr__

        mutated = report.model_copy(update={"program_key": Hostile()})
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "reports.db")
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises((ValueError, TypeError)) as raised:
                    await store.publish_report(
                        mutated, publisher_id="publisher", redact_json=lambda value: value
                    )
            captured = capsys.readouterr()
            diagnostics = (
                str(raised.value)
                + repr(raised.value)
                + captured.out
                + captured.err
                + caplog.text
                + str([str(item.message) for item in caught])
            )
            assert "MUTATED-REPORT-PRIVATE-CANARY" not in diagnostics
        finally:
            await store.close()

    asyncio.run(exercise())


async def _structured_output_conformance(store):
    from cayu.evals.read_views import EvalViewQuery
    from cayu.evals.reported import EvalReport, EvalReportConflict

    payload = await _payload("finite-target")
    report = EvalReport.model_validate(payload)
    for index, value in enumerate((float("nan"), float("inf"), float("-inf"))):
        for nested in ({"value": value}, [None, {"value": [value]}]):
            invalid = report.model_copy(deep=True)
            # Public stores must reject invalid post-construction values too.
            object.__setattr__(invalid, "report_id", f"nonfinite-{index}")
            object.__setattr__(invalid.evidence.run.cases[0].trials[0], "structured_output", nested)
            with pytest.raises((TypeError, ValueError)):
                await store.publish_report(
                    invalid, publisher_id="finite-publisher", redact_json=lambda value: value
                )
    assert not (await store.list_view_index(EvalViewQuery(target_key="finite-target"))).items
    valid = report.model_copy(deep=True)
    structured = {"values": [None, True, False, 1, 0, 1.25, -2.5]}
    object.__setattr__(valid.evidence.run.cases[0].trials[0], "structured_output", structured)
    published = await store.publish_report(
        valid, publisher_id="finite-publisher", redact_json=lambda value: value
    )
    assert await store.load_report(published.receipt.id) == published
    assert (
        await store.publish_report(
            valid, publisher_id="finite-publisher", redact_json=lambda value: value
        )
        == published
    )
    changed = valid.model_copy(deep=True)
    changed.evidence.run.cases[0].trials[0].structured_output["values"][1] = 1
    with pytest.raises(EvalReportConflict):
        await store.publish_report(
            changed, publisher_id="finite-publisher", redact_json=lambda value: value
        )
    return published


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_report_structured_output_is_finite_before_any_publication(tmp_path, backend):
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    async def exercise():
        path = tmp_path / "finite.db"
        store = InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(path)
        try:
            published = await _structured_output_conformance(store)
        finally:
            await store.close()
        if backend == "sqlite":
            reopened = SQLiteEvalStore(path)
            try:
                assert await reopened.load_report(published.receipt.id) == published
                assert (
                    await reopened.publish_report(
                        published.report,
                        publisher_id="finite-publisher",
                        redact_json=lambda value: value,
                    )
                    == published
                )
            finally:
                await reopened.close()

    asyncio.run(exercise())


def test_postgres_report_structured_output_is_finite_before_any_publication(postgres_dsn):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await _structured_output_conformance(store)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_report_store_revalidates_trial_summaries_before_mutation(tmp_path, backend):
    from tests.server.test_server_eval_reporting import _trial_summary_payloads

    from cayu.evals.read_views import EvalViewQuery
    from cayu.evals.reported import EvalReport, report_operation_id
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    async def exercise():
        path = tmp_path / "trial-summary-store.db"
        store = InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(path)
        retained = []
        try:
            base = await _payload("ci-target")
            for name, payload, valid in sorted(
                _trial_summary_payloads(base), key=lambda item: item[2]
            ):
                # Bypass construction, as callers can mutate an existing typed object.
                candidate = EvalReport.model_construct(**payload)
                if not valid:
                    with pytest.raises(ValueError):
                        await store.publish_report(
                            candidate, publisher_id="publisher", redact_json=lambda x: x
                        )
                    assert await store.load_report(report_operation_id("publisher", name)) is None
                    assert not (
                        await store.list_view_index(EvalViewQuery(target_key="ci-target"))
                    ).items
                else:
                    accepted = await store.publish_report(
                        candidate, publisher_id="publisher", redact_json=lambda x: x
                    )
                    assert await store.load_report(accepted.receipt.id) == accepted
                    assert (
                        await store.publish_report(
                            candidate, publisher_id="publisher", redact_json=lambda x: x
                        )
                        == accepted
                    )
                    retained.append(accepted)
        finally:
            await store.close()
        if backend == "sqlite":
            reopened = SQLiteEvalStore(path)
            try:
                for record in retained:
                    assert await reopened.load_report(record.receipt.id) == record
            finally:
                await reopened.close()

    asyncio.run(exercise())
