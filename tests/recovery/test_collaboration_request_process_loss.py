"""Exact public control replay after process loss on both persistent owners."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestCommand, RequestControl
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

pytestmark = [pytest.mark.anyio, pytest.mark.process]


@pytest.mark.parametrize(
    "backend",
    [
        "sqlite",
        pytest.param("postgres", marks=[pytest.mark.postgres, pytest.mark.postgres_recovery]),
    ],
)
async def test_answer_commit_survives_process_loss_and_fresh_process_replay(
    backend, tmp_path, request
):
    address = (
        str(tmp_path / "answer.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    output = tmp_path / "answer.json"
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        CAYU_REQUEST_TEST_STORE=address,
        PYTHONPATH=str(root / "src") + os.pathsep + str(root),
    )
    for mode, expected_code in (("publish", 19), ("recover", 0)):
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.recovery.collaboration_answer_worker",
                backend,
                mode,
                str(output),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert child.returncode == expected_code, child.stderr.decode()
        if mode == "recover":
            assert b"answer-replay-ok" in child.stdout


@pytest.mark.parametrize(
    "backend",
    [
        "sqlite",
        pytest.param("postgres", marks=[pytest.mark.postgres, pytest.mark.postgres_recovery]),
    ],
)
async def test_control_process_loss_replays_without_another_acceptance(backend, tmp_path, request):
    address = (
        str(tmp_path / "request.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    output = tmp_path / "accepted.json"
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        CAYU_REQUEST_TEST_STORE=address,
        PYTHONPATH=str(root / "src") + os.pathsep + str(root),
    )
    child = subprocess.run(
        [sys.executable, "-m", "tests.recovery.collaboration_request_worker", backend, str(output)],
        cwd=root,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 19, child.stderr.decode()
    saved = json.loads(output.read_text(encoding="utf-8"))
    expected = prepare_contract(RequestCommand, saved["expected"], redactor=SecretRedactor())
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    resolver = RequestResolver(expected.intent.request)
    application = app(
        store,
        registration(scope=saved["binding"]["application_scope"]),
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    try:
        initialized = await application.initialize_collaboration()
        before = await application.inspect_collaboration_request(expected, context=resolver.context)
        assert before.state == "cancelled"
        control = RequestControl(
            operation=initialized.operation("control"),
            expected=expected,
            expected_revision=1,
            kind="cancel",
        )
        replay = await application.control_collaboration_request(control, context=resolver.context)
        assert replay == before.terminal
        assert (
            await application.inspect_collaboration_request(expected, context=resolver.context)
            == before
        )
        participant = await application.inspect_participant(
            expected.intent.selection.recipient.reference, context=CONTEXT
        )
        assert participant.issued_permit_frontier == 1
        assert participant.outstanding_obligations == 0
    finally:
        await application.drain_collaboration_requests()
        await store.close()
