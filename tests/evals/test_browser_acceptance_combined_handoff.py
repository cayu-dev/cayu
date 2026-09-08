"""Combined scorecard evidence; not a replacement for a live authenticated trial."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.evals.test_browser_acceptance_operator_oracle import _case, _settled_record

from cayu.evals.browser_acceptance import (
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceDiagnosticV1,
    _case_operator_evidence,
    _project_operator_record,
    _semantic_state,
    _validated_operator_observation_revision,
)
from cayu.evals.corpus import _content_revision
from cayu.runtime.browser_control import BrowserControlCheckpoint, BrowserControlRecord
from cayu.tools.browser_session import _browser_operation_id_sha256, _durable_browser_operation_key


def _observation(record):
    return {
        "record_type": "cayu.browser-operation",
        "schema_version": 1,
        "state": "terminal",
        "operation": "observe",
        "operation_id_sha256": _browser_operation_id_sha256("fresh"),
        "parent_session_id": record.identity.session_id,
        "parent_run_epoch": record.identity.run_epoch,
        "execution_profile_fingerprint": record.identity.execution_profile_fingerprint,
        "environment_name": record.identity.environment_name,
        "allocation_fingerprint": record.identity.allocation_fingerprint,
        "browser_session_id": record.identity.browser_session_id,
        "invocation_control_epoch": record.control_epoch,
        "observation_confirmed": True,
        "observation_protected": True,
    }


@pytest.mark.parametrize("field", list(_observation(_settled_record())))
def test_observation_digest_requires_every_authority_field(field):
    record = _settled_record()
    operation = _observation(record)
    assert _validated_operator_observation_revision(operation, "fresh", record).startswith(
        "sha256:"
    )
    operation[field] = None
    with pytest.raises(ValueError, match="another control boundary"):
        _validated_operator_observation_revision(operation, "fresh", record)


@pytest.mark.parametrize("field", ["observation_confirmed", "observation_protected"])
def test_observation_flags_require_positive_boolean_evidence(field):
    record = _settled_record()
    operation = _observation(record)
    operation[field] = 1
    with pytest.raises(ValueError):
        _validated_operator_observation_revision(operation, "fresh", record)


@pytest.mark.parametrize("restored_state", ["closed", "control_uncertain"])
def test_combined_operator_projection_reads_both_serialized_allocation_records(restored_state):
    first = _settled_record()
    second = BrowserControlRecord(
        identity=first.identity.model_copy(update={"browser_session_id": "bs_restored"}),
        state=restored_state,
    )
    checkpoint = BrowserControlCheckpoint(records=(first, second))
    observations = {}
    for operation_id, record in (("fresh", first), ("restored", second)):
        value = _observation(record)
        value["operation_id_sha256"] = _browser_operation_id_sha256(operation_id)
        observations[_durable_browser_operation_key(operation_id)] = json.dumps(value)

    class ReconstructedStore:
        async def load_checkpoint(self, session_id):
            assert session_id == first.identity.session_id
            return {"browser_controls": json.loads(checkpoint.model_dump_json())}

        async def load_session_operation(self, session_id, key):
            assert session_id == first.identity.session_id
            return json.loads(observations[key])

    async def scenario():
        projected = await _case_operator_evidence(
            SimpleNamespace(session_store=ReconstructedStore()),
            SimpleNamespace(
                oracle_parameters={
                    "required_operator_inputs": 2,
                    "required_operator_observation_id": "fresh",
                    "required_restored_observation_id": "restored",
                }
            ),
            SimpleNamespace(
                session_id=first.identity.session_id,
                trajectory=SimpleNamespace(
                    session=SimpleNamespace(
                        id=first.identity.session_id,
                        instance_id=first.identity.session_instance_id,
                    )
                ),
            ),
            expected_execution_profile_fingerprint=first.identity.execution_profile_fingerprint,
        )
        assert projected.fresh_observation_revision is not None
        assert projected.restored_observation_revision is not None
        assert projected.restored_browser_session_revision != projected.browser_session_revision

    if restored_state == "closed":
        asyncio.run(scenario())
    else:
        with pytest.raises(ValueError, match="distinct closed allocation"):
            asyncio.run(scenario())


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "operator",
        "profile",
        "authentication",
        "restored",
        "same_browser",
        "same_observation",
        "wrong_target",
        "unclosed",
        "first_only",
        "swapped_phase",
        "phase_identity",
    ],
)
def test_combined_handoff_and_restoration_survive_json_without_weakening_oracles(missing):
    operations = ("navigate", "observe", "close") * 2
    source = _case()
    case = BrowserAcceptanceCaseV1.build(
        **{
            **source.model_dump(mode="python", exclude={"revision"}),
            "operations": operations,
            "oracle_parameters": {
                "required_operations": list(operations),
                "required_operator_inputs": 2,
                "required_restored_observation_id": "restored",
                "required_distinct_browser_sessions": 2,
                "required_profile_checkpoint_delta": 2,
                "required_authenticated_requests": 2,
                "required_protected_observation_target": "https://site.test/member",
            },
        }
    )
    operator = _project_operator_record(_settled_record()).model_dump(mode="json")
    first, second = operator["browser_session_revision"], "sha256:" + "f" * 64
    operator.update(
        fresh_observation_revision="sha256:" + "a" * 64,
        restored_observation_revision="sha256:" + "b" * 64,
        restored_browser_session_revision=second,
    )
    if missing == "restored":
        operator["restored_observation_revision"] = None
    if missing == "same_browser":
        operator["restored_browser_session_revision"] = first
    if missing == "same_observation":
        operator["restored_observation_revision"] = operator["fresh_observation_revision"]
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state="captured",
        operator=None if missing == "operator" else operator,
        profile=None
        if missing == "profile"
        else {
            "store_kind": "sqlite",
            "status": "available",
            "authority_fingerprint": "1" * 64,
            "generation_before": 0,
            "generation_after": 2,
            "active_writer": False,
            "cookie_count": 1,
            "checkpoint_receipt_revision": "sha256:" + "c" * 64,
            "restore_receipt_revision": "sha256:" + "d" * 64,
        },
        fixture_authenticated_request_count=None if missing == "authentication" else 2,
        authentication_phases=(
            {
                "phase": "handback",
                "browser_session_revision": first,
                "observation_revision": operator["fresh_observation_revision"],
                "authenticated_requests": 2 if missing == "first_only" else 1,
            },
            {
                "phase": "handback" if missing == "swapped_phase" else "restoration",
                "browser_session_revision": first if missing == "phase_identity" else second,
                "observation_revision": "sha256:" + "b" * 64,
                "authenticated_requests": 0 if missing == "first_only" else 1,
            },
        ),
        operations=tuple(
            {
                "sequence": index + 1,
                "invocation_revision": "sha256:" + str(index + 1) * 64,
                "operation": operation,
                "state": "terminal",
                "allocation_disposition": "retired"
                if operation == "close" and missing != "unclosed"
                else "live",
                "browser_session_revision": first if index < 3 else second,
                "observed_target_revision": _content_revision(
                    {
                        "url": "https://site.test/login"
                        if missing == "wrong_target"
                        else "https://site.test/member"
                    },
                    "browser acceptance observed target",
                ),
            }
            for index, operation in enumerate(operations)
        ),
    )
    reconstructed = BrowserAcceptanceDiagnosticV1.model_validate_json(diagnostic.model_dump_json())
    assert _semantic_state(case, reconstructed, public_operations=frozenset()).value == (
        "passed" if missing is None else "failed"
    )
