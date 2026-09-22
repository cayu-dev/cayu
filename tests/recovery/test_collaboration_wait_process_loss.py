"""Fresh-process wait reconstruction after registration acknowledgement loss."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import app, registration

from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestCommand, RequestControl
from cayu.collaboration.waits import CollaborationWait
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
async def test_wait_registration_reopens_after_process_loss(backend, tmp_path, request):
    address = (
        str(tmp_path / "wait.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    output = tmp_path / "wait.json"
    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "src") + os.pathsep + str(root),
    )
    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.recovery.collaboration_wait_worker",
            backend,
            address,
            str(output),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        timeout=60,
    )
    assert worker.returncode == 19, worker.stderr.decode()
    saved = json.loads(output.read_text(encoding="utf-8"))
    expected = prepare_contract(RequestCommand, saved["expected"], redactor=SecretRedactor())
    wait = CollaborationWait.model_validate(saved["wait"])
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    resolver = RequestResolver(expected.intent.request)
    application = app(
        store,
        registration(scope=saved["scope"]),
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    try:
        await application.initialize_collaboration()
        control = RequestControl(
            operation=wait.operation.model_copy(update={"caller_key": "wait-process-control"}),
            expected=expected,
            expected_revision=1,
            kind="cancel",
        )
        await application.control_collaboration_request(control, context=resolver.context)
        elected = await application.observe_collaboration_wait(wait, context=resolver.context)
        assert elected.state == "elected"
        assert elected.election is not None
        assert elected.election.result == "failure"
        assert len(elected.source_pins) == 0
    finally:
        await application.drain_collaboration_requests()
        await store.close()
