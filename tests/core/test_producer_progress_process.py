"""Public progress replay uses retained source evidence across real process boundaries."""

import asyncio
import json
import sys

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ProducerProgressOccurrence
from cayu.collaboration.requests import RequestProgressReceipt


@pytest.mark.anyio
@pytest.mark.parametrize("native_stores", ["sqlite", "postgres"], indirect=True)
async def test_progress_fresh_process_exact_replay_without_native_owner(native_stores, monkeypatch):
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, _, initialized, registration, _ = values
    prior = await app.inspect_collaboration_request(
        admission.expected, context=resolver.recipient.context
    )
    occurrence = ProducerProgressOccurrence(
        operation=initialized.operation("fresh-process-progress"),
        expected_revision=prior.revision,
        sequence=1,
        kind="published",
    )
    expected = await app.record_producer_progress(
        registration, occurrence, context=resolver.recipient.context
    )
    backend, address = native_stores[3]
    material = json.dumps(
        {
            "backend": backend,
            "address": address,
            "expected": registration.model_dump(mode="json"),
            "occurrence": occurrence.model_dump(mode="json"),
        }
    ).encode()
    for _ in range(2):
        reader = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.producer_progress_reader_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(reader.communicate(material), 90)
            assert reader.returncode == 0, error.decode()
            assert RequestProgressReceipt.model_validate_json(output) == expected
        finally:
            if reader.returncode is None:
                reader.kill()
                await reader.wait()
    after = await app.inspect_collaboration_request(
        admission.expected, context=resolver.recipient.context
    )
    assert after.revision == prior.revision + 1 and len(after.progress) == 1
    assert len(provider.requests) == 1
