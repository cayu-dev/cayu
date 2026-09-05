"""Application identity survives the numeric representations of durable stores."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from pydantic import ValidationError
from tests.evals.test_corpus_execution import _corpus, _provider, _target
from tests.evals.test_sqlite_eval_store import _request

from cayu import CayuApp, CayuConfig, RunDefaults, ToolExecutionConfig
from cayu.evals import EvaluationTargetIdentity
from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_reporting import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
)
from cayu.runtime.manifest import _app_manifest_fingerprint
from cayu.runtime.retry_policy import RetryPolicy
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

_CONFIGS = [
    None,
    CayuConfig(
        tool_execution=ToolExecutionConfig(tool_timeout_seconds=42.5),
        run=RunDefaults(retry_policy=RetryPolicy(initial_delay_s=3.0)),
    ),
]


def _normalize_numbers(source: str):
    return json.loads(
        source,
        parse_float=lambda text: int(float(text)) if float(text).is_integer() else float(text),
    )


@pytest.mark.parametrize("config", _CONFIGS, ids=["default", "explicit"])
def test_target_identity_survives_numeric_normalization(config):
    target = EvaluationTargetIdentity(
        target_key="probe",
        application_release_id="probe-v1",
        app_manifest=CayuApp(config=config, enable_logging=False).describe(),
    )
    assert EvaluationTargetIdentity.model_validate_json(target.model_dump_json()) == target
    normalized = _normalize_numbers(target.model_dump_json())
    assert EvaluationTargetIdentity.model_validate_json(json.dumps(normalized)) == target
    normalized["app_manifest"]["runtime"]["configuration"]["values"]["tool_execution"][
        "tool_timeout_seconds"
    ] = 43.5
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        EvaluationTargetIdentity.model_validate_json(json.dumps(normalized))


def test_nested_untyped_numeric_values_keep_identity_but_real_changes_do_not():
    payload = CayuApp(enable_logging=False).describe().model_dump(mode="json")
    payload["runtime"]["configuration"]["values"]["nested"] = {
        "numbers": [30.0, -0.0, 1e18, 0.125, 1e-7, {"value": 2.0}],
        "boolean": True,
    }
    payload["fingerprint"] = _app_manifest_fingerprint(payload)
    target = EvaluationTargetIdentity(
        target_key="probe", application_release_id="probe-v1", app_manifest=payload
    )
    normalized = _normalize_numbers(target.model_dump_json())
    assert EvaluationTargetIdentity.model_validate_json(json.dumps(normalized)) == target
    for value in (False, 1, "true"):
        changed = json.loads(json.dumps(normalized))
        changed["app_manifest"]["runtime"]["configuration"]["values"]["nested"]["boolean"] = value
        with pytest.raises(ValidationError, match="fingerprint does not match"):
            EvaluationTargetIdentity.model_validate_json(json.dumps(changed))


@pytest.mark.parametrize("normalized", [False, True])
def test_schema_16_identity_is_explicitly_incompatible(normalized):
    # Reconstruct the previous writer's exact algorithm, without blessing its hash.
    manifest = CayuApp(enable_logging=False).describe().model_dump(mode="json")
    manifest.pop("fingerprint")
    manifest["schema_version"] = "16"
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    document = json.dumps(
        dict(target_key="probe", application_release_id="probe-v1", app_manifest=manifest)
    )
    if normalized:
        document = json.dumps(_normalize_numbers(document))
    with pytest.raises(ValidationError, match="Input should be '17'"):
        EvaluationTargetIdentity.model_validate_json(document)


async def _result(config):
    corpus = _corpus(trials=1)
    result = await run_corpus_suite(
        _target(_provider(trials=1), config=config),
        corpus,
        corpus.suites[0].id,
        max_concurrency=1,
    )
    return corpus, result


@pytest.mark.parametrize("config", _CONFIGS, ids=["default", "explicit"])
def test_corpus_result_public_json_round_trip(config):
    _, result = asyncio.run(_result(config))
    encoded = corpus_execution_result_to_json(result)
    assert corpus_execution_result_from_json(encoded) == result
    assert corpus_execution_result_from_json(json.dumps(_normalize_numbers(encoded))) == result


@pytest.mark.parametrize("normalized", [False, True])
def test_saved_schema_16_result_is_rejected_without_rehashing(normalized):
    _, result = asyncio.run(_result(None))
    document = result.model_dump(mode="json")
    manifest = document["target"]["app_manifest"]
    manifest.pop("fingerprint")
    manifest["schema_version"] = "16"
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    encoded = json.dumps(document)
    if normalized:
        encoded = json.dumps(_normalize_numbers(encoded))
    with pytest.raises(ValidationError, match="Input should be '17'"):
        corpus_execution_result_from_json(encoded)


async def _save_result(store, corpus, result):
    redact_json = SecretRedactor().redact_json
    await store.save_corpus(corpus, redact_json=redact_json)
    await store.admit_run(_request(corpus), redact_json=redact_json)
    lease = await store.claim_run()
    assert lease is not None
    await store.publish_result(lease.claim, result, redact_json=redact_json)
    return lease.claim.run_id


@pytest.mark.parametrize("config", _CONFIGS, ids=["default", "explicit"])
def test_sqlite_result_numeric_identity_after_reopen(tmp_path, config):
    async def exercise():
        corpus, result = await _result(config)
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(path)
        try:
            run_id = await _save_result(store, corpus, result)
        finally:
            await store.close()
        reopened = SQLiteEvalStore(path)
        try:
            assert await reopened.load_result(run_id) == result
            assert await reopened.load_result_by_revision(result.revision) == result
        finally:
            await reopened.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("config", _CONFIGS, ids=["default", "explicit"])
def test_postgres_result_numeric_identity_after_reopen(postgres_dsn, config):
    import psycopg

    from cayu.storage.evals_postgres import PostgresEvalStore

    async def exercise():
        corpus, result = await _result(config)
        # Exercise actual JSONB number decoding, independently of the store's copy.
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            row = await (
                await conn.execute("SELECT %s::jsonb", (result.model_dump_json(),))
            ).fetchone()
            assert row is not None
            assert corpus_execution_result_from_json(json.dumps(row[0])) == result
        from tests.evals.test_postgres_eval_store import _drop_eval_tables

        await _drop_eval_tables(postgres_dsn)
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            run_id = await _save_result(store, corpus, result)
        finally:
            await store.close()
        reopened = PostgresEvalStore(postgres_dsn)
        try:
            assert await reopened.load_result(run_id) == result
            assert await reopened.load_result_by_revision(result.revision) == result
        finally:
            await reopened.close()

    asyncio.run(exercise())
