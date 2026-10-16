"""Projection preparation contracts independent of event and transcript adapters."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cayu.events import Event, EventType, event_with_runtime_nested_payload_authority
from cayu.messages import Message, ToolResultPart
from cayu.runtime._event_projection import (
    prepare_new_runtime_event,
    project_persisted_runtime_event,
    project_runtime_event,
)
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
    redact_untrusted_message_for_boundary,
)
from cayu.runtime._tool_result_projection import (
    PreparedToolResultProjection,
    ValidatedToolResultMessage,
    read_runtime_tool_result_message,
    read_runtime_tool_result_projection,
)
from cayu.tools.base import ToolResult
from cayu.tools.result_projection import MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def projection_payload(status="externalized", content="private-preview", method=None):
    artifact_id = "art_" + "a" * 32
    reference = {
        "type": "cayu.tool_result_artifact.v1",
        "artifact_id": artifact_id,
        "store_id": "artifacts",
        "filename": f"tool-result-{artifact_id}.txt",
        "content_type": "text/plain; charset=utf-8",
        "size_bytes": 1000,
        "sha256": "b" * 64,
        "scope": "session",
        "session_id_sha256": "c" * 64,
        "projection_authority": "cayu.tool_result_projection.v1",
        "readback_max_bytes": 64,
    }
    record = {
        "schema_version": 1,
        "status": status,
        "policy_id": "test.projection.v1",
        "original_bytes": len(content.encode("utf-8")),
        "projected_bytes": len(content.encode("utf-8")),
        "original_token_estimate": len(content),
        "projected_token_estimate": len(content),
        "token_estimation_method": method or "unicode_codepoints_divided_by_4_ceiling_v1",
    }
    artifacts = [{"application": {"value": "private-preview"}}]
    if status == "externalized":
        record.update(
            artifact_id=artifact_id,
            artifact_sha256=reference["sha256"],
            logical_identity_sha256="d" * 64,
            tool_call_id_sha256="e" * 64,
        )
        artifacts.append(reference)
    elif status == "failed":
        record["failure_type"] = "artifact_store_failure"
    return {
        "result": ToolResult(
            content=content, artifacts=artifacts, structured={"receipt": "retained"}, is_error=True
        ).model_dump(mode="json"),
        "tool_result_projection": record,
    }


@pytest.mark.parametrize("status", ["unchanged", "failed", "externalized"])
@pytest.mark.parametrize(
    "method", ["unicode_codepoints_divided_by_7_ceiling_v1", "custom.estimator.v1"]
)
@settings(max_examples=30, deadline=None, derandomize=True, database=None)
@given(preview=st.text(alphabet="abcdefé界🙂\n", max_size=100))
def test_preparation_preserves_identity_and_accounts_after_redaction(status, method, preview):
    payload = projection_payload(status, preview + "private-preview", method)
    before = deepcopy(payload)
    source = read_runtime_tool_result_projection(payload)
    assert source is not None
    redactor = SecretRedactor(["private-preview", "art_"])
    redacted = redactor.redact_json(payload["result"])
    prepared = PreparedToolResultProjection(source, redacted_result=redacted, redactor=redactor)
    record = prepared.finish(prepared.content).record_payload()
    assert "private-preview" not in prepared.content
    assert prepared.artifacts[0]["application"]["value"] == REDACTED_SECRET
    assert record["projected_bytes"] == len(prepared.content.encode("utf-8"))
    if method.startswith("unicode_"):
        assert record["projected_token_estimate"] == (len(prepared.content) + 6) // 7
        assert record["token_estimation_method"] == method
    else:
        assert record["projected_token_estimate"] == 0
        assert record["token_estimation_method"] == "unavailable_after_boundary_redaction_v1"
    if status == "externalized":
        assert prepared.artifacts[1] == before["result"]["artifacts"][1]
        assert prepared.content.count("[cayu tool-result projection reference]") == 1
        assert before["tool_result_projection"]["artifact_id"] in prepared.content
    else:
        assert "[cayu tool-result projection reference]" not in prepared.content
    assert len(prepared.content.encode("utf-8")) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert payload == before
    assert redacted == redactor.redact_json(before["result"])

    # A fresh process only retains serialized result/evidence, not this value.
    replay_payload = deepcopy(before)
    replay_payload["result"].update(content=prepared.content, artifacts=prepared.artifact_payload())
    replay_payload["tool_result_projection"] = prepared.finish(prepared.content).record_payload()
    replay = read_runtime_tool_result_projection(replay_payload)
    assert replay is not None
    repeated = PreparedToolResultProjection(
        replay, redacted_result=redactor.redact_json(replay_payload["result"]), redactor=redactor
    )
    assert repeated.content == prepared.content
    assert repeated.artifacts == prepared.artifacts
    assert repeated.finish(repeated.content).record_payload() == record


@pytest.mark.parametrize("damage", ["extra", "missing", "authority", "hash", "count", "duplicate"])
def test_invalid_projection_evidence_never_produces_a_validated_value(damage):
    payload = projection_payload()
    reference = payload["result"]["artifacts"][1]
    if damage == "extra":
        reference["unexpected"] = True
    elif damage == "missing":
        reference.pop("readback_max_bytes")
    elif damage == "authority":
        reference["projection_authority"] = "application-owned"
    elif damage == "hash":
        reference["sha256"] = "f" * 64
    elif damage == "count":
        payload["tool_result_projection"]["projected_bytes"] += 1
    else:
        payload["result"]["artifacts"].append(deepcopy(reference))
    assert read_runtime_tool_result_projection(payload) is None


@pytest.mark.parametrize("field", ["result", "tool_result_projection"])
def test_invalid_projection_containers_preserve_boundary_errors(field):
    payload = projection_payload()
    payload[field] = []
    with pytest.raises(TypeError, match="object result and record"):
        read_runtime_tool_result_projection(payload)


def test_preparation_values_and_payload_copies_do_not_share_mutable_state():
    payload = projection_payload()
    source = read_runtime_tool_result_projection(payload)
    assert source is not None
    prepared = PreparedToolResultProjection(
        source, redacted_result=payload["result"], redactor=SecretRedactor()
    )
    payload["result"]["artifacts"][1]["artifact_id"] = "changed"
    payload["tool_result_projection"]["policy_id"] = "changed"
    reference = source.references()[1]
    reference["artifact_id"] = "changed"
    emitted = prepared.artifact_payload()
    emitted[0]["application"]["value"] = "changed"
    emitted_record = prepared.finish("final content").record_payload()
    emitted_record["policy_id"] = "changed"
    assert source.references()[1]["artifact_id"] == "art_" + "a" * 32
    assert prepared.artifacts[0]["application"]["value"] == "private-preview"
    assert prepared.finish(prepared.content).record_payload()["policy_id"] == "test.projection.v1"
    with pytest.raises(FrozenInstanceError):
        prepared.content = "changed"
    with pytest.raises(TypeError):
        prepared.artifacts[0]["application"]["value"] = "changed"
    assert "private-preview" not in repr(source) + repr(prepared)


@pytest.mark.parametrize("characters", ["x", "é", "界", "🙂"])
def test_externalized_preparation_bounds_expanding_redaction_before_accounting(characters):
    payload = projection_payload(content=characters * 16000)
    source = read_runtime_tool_result_projection(payload)
    assert source is not None
    redactor = SecretRedactor(characters)
    prepared = PreparedToolResultProjection(
        source, redacted_result=redactor.redact_json(payload["result"]), redactor=redactor
    )
    record = prepared.finish(prepared.content).record_payload()
    assert record["projected_bytes"] == len(prepared.content.encode("utf-8"))
    assert record["projected_bytes"] <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert prepared.content.endswith("use only advertised offsets.")
    assert record["projected_token_estimate"] == (len(prepared.content) + 3) // 4


def test_final_accounting_uses_content_after_other_terminal_policies():
    payload = projection_payload("failed")
    source = read_runtime_tool_result_projection(payload)
    assert source is not None
    prepared = PreparedToolResultProjection(
        source, redacted_result=payload["result"], redactor=SecretRedactor()
    )
    final = "replacement 🙂"
    record = prepared.finish(final).record_payload()
    assert record["projected_bytes"] == len(final.encode("utf-8"))
    assert record["projected_token_estimate"] == (len(final) + 3) // 4
    assert record["status"] == "failed"
    assert prepared.content == "private-preview"


def test_preparation_rejects_raw_projection_records():
    payload = projection_payload("unchanged")
    with pytest.raises(TypeError, match="requires a validated projection value"):
        PreparedToolResultProjection(
            payload["tool_result_projection"],
            redacted_result=payload["result"],
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize("status", ["unchanged", "failed"])
@pytest.mark.parametrize("persisted", [False, True])
def test_non_externalized_event_preserves_omitted_optional_result_fields(status, persisted):
    payload = projection_payload(status)
    del payload["result"]["artifacts"]
    del payload["result"]["structured"]
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    if persisted:
        output = project_persisted_runtime_event(event, sequence=1, redactor=SecretRedactor())
    else:
        output = prepare_new_runtime_event(event, redactor=SecretRedactor())
    assert output.payload["result"] == {"content": "private-preview", "is_error": True}
    assert output.payload["tool_result_projection"]["projected_bytes"] == len("private-preview")


@pytest.mark.parametrize("status", ["unchanged", "failed"])
def test_non_externalized_event_with_absent_content_preserves_boundary_error(status):
    payload = projection_payload(status)
    payload["result"] = {}
    payload["tool_result_projection"]["projected_bytes"] = 0
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    with pytest.raises(AssertionError, match="^Validated projection lost its content.$"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


@pytest.mark.parametrize("status", ["unchanged", "failed"])
@pytest.mark.parametrize(
    "malformed_control", ["effect", "workspace_mutation_capture_status", "registration_state"]
)
@pytest.mark.parametrize("project", [project_runtime_event, project_persisted_runtime_event])
def test_malformed_replay_preserves_inline_result_without_content(
    status, malformed_control, project
):
    payload = projection_payload(status, content="")
    payload["result"] = {}
    payload[malformed_control] = "bogus"
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    output = project(event, sequence=1, redactor=SecretRedactor())
    assert output.payload["result"] == {}
    assert output.payload["tool_result_projection"] == payload["tool_result_projection"]
    # New writes retain strict control validation; only exposure degrades.
    with pytest.raises(ValueError):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


@pytest.mark.parametrize(
    ("project", "malformed_effect"),
    [
        (prepare_new_runtime_event, False),
        (project_runtime_event, False),
        (project_persisted_runtime_event, False),
        (project_runtime_event, True),
        (project_persisted_runtime_event, True),
    ],
)
@pytest.mark.parametrize("content_present", [False, True])
def test_externalized_event_preserves_missing_and_empty_content(
    project, malformed_effect, content_present
):
    payload = projection_payload(content="")
    if not content_present:
        del payload["result"]["content"]
    if malformed_effect:
        payload["effect"] = "bogus"
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    if not content_present:
        with pytest.raises(
            AssertionError, match="^Validated projection lost its content or artifacts.$"
        ):
            project(event, redactor=SecretRedactor(), **kwargs)
        return
    output = project(event, redactor=SecretRedactor(), **kwargs)
    assert output.payload["result"]["content"].count("[cayu tool-result projection reference]") == 1
    assert output.payload["result"]["artifacts"][1] == payload["result"]["artifacts"][1]


@pytest.mark.parametrize(
    "malformed_control", ["effect", "workspace_mutation_capture_status", "registration_state"]
)
@pytest.mark.parametrize("project", [project_runtime_event, project_persisted_runtime_event])
def test_malformed_replay_still_restores_valid_externalized_reference(malformed_control, project):
    payload = projection_payload()
    payload[malformed_control] = "bogus"
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    output = project(event, sequence=1, redactor=SecretRedactor(["private-preview", "art_"]))
    result = output.payload["result"]
    assert "private-preview" not in result["content"]
    assert result["content"].count("[cayu tool-result projection reference]") == 1
    assert result["artifacts"][1] == payload["result"]["artifacts"][1]


@pytest.mark.parametrize("status", ["unchanged", "failed", "externalized"])
@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
def test_event_accounting_describes_content_after_later_result_policy(monkeypatch, status, project):
    from cayu.runtime import _event_projection as events

    payload = projection_payload(status)
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    restore = events.restore_attested_event_result
    observed_prepared_content = []
    final_content = "later result policy 🙂"

    def replace_after_preparation(event, *, redacted_payload, trust_persisted, reject_malformed):
        restore(
            event,
            redacted_payload=redacted_payload,
            trust_persisted=trust_persisted,
            reject_malformed=reject_malformed,
        )
        observed_prepared_content.append(redacted_payload["result"]["content"])
        redacted_payload["result"]["content"] = final_content

    # Control a later restoration stage to exercise the actual event pipeline.
    monkeypatch.setattr(events, "restore_attested_event_result", replace_after_preparation)
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    output = project(event, redactor=SecretRedactor("private-preview"), **kwargs)
    assert len(observed_prepared_content) == 1
    assert "private-preview" not in observed_prepared_content[0]
    if status == "externalized":
        assert "[cayu tool-result projection reference]" in observed_prepared_content[0]
    assert output.payload["result"]["content"] == final_content
    record = output.payload["tool_result_projection"]
    assert record["status"] == status
    assert record["projected_bytes"] == len(final_content.encode("utf-8"))
    assert record["projected_token_estimate"] == (len(final_content) + 3) // 4
    assert event.payload == payload


@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
@pytest.mark.parametrize("secrets", [["private-preview"], ["art_"], ["private-preview", "art_"]])
def test_event_and_transcript_adapters_prepare_equivalent_results(project, secrets):
    payload = projection_payload(content="private-preview 界🙂")
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    message = Message.tool_result(
        results=[ToolResultPart(tool_call_id="call", tool_name="tool", **payload["result"])]
    )
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    event_result = project(event, redactor=SecretRedactor(secrets), **kwargs).payload["result"]
    transcript = redact_runtime_message_for_boundary(
        message, redactor=SecretRedactor(secrets), field_name="transcript"
    )
    part = transcript.content[0].model_dump(mode="json")
    assert {name: part[name] for name in payload["result"]} == event_result
    repeated = redact_runtime_message_for_boundary(
        transcript, redactor=SecretRedactor(secrets), field_name="transcript"
    )
    assert repeated == transcript
    assert message.content[0].content == "private-preview 界🙂"


def test_projection_preparation_does_not_authorize_untrusted_message_ingress():
    payload = projection_payload()
    message = Message.tool_result(
        results=[ToolResultPart(tool_call_id="call", tool_name="tool", **payload["result"])]
    )
    redactor = SecretRedactor(["private-preview", "art_"])
    projected = redact_untrusted_message_for_boundary(
        message, redactor=redactor, field_name="ingress"
    )
    part = projected.content[0]
    assert part.content == REDACTED_SECRET
    assert "projection_authority" not in part.artifacts[1]
    assert part.artifacts[1]["artifact_id"] != payload["result"]["artifacts"][1]["artifact_id"]
    assert "[cayu tool-result projection reference]" not in part.content


def test_validated_message_references_are_detached_even_when_constructed_directly():
    payload = projection_payload()
    artifacts = payload["result"]["artifacts"]
    source = ValidatedToolResultMessage("preview", artifacts)
    artifacts[1]["artifact_id"] = "changed"
    copied = source.references()
    with pytest.raises(TypeError):
        copied[1]["artifact_id"] = "changed-again"
    with pytest.raises(TypeError):
        copied[1] = {}
    assert source.references()[1]["artifact_id"] == "art_" + "a" * 32
    with pytest.raises(FrozenInstanceError):
        source.original_content = "changed"


@pytest.mark.parametrize("damage", ["extra", "hash", "authority", "duplicate"])
def test_transcript_adapter_does_not_reconstruct_invalid_or_ambiguous_references(damage):
    payload = projection_payload()
    artifacts = payload["result"]["artifacts"]
    if damage == "extra":
        artifacts[1]["extra"] = True
    elif damage == "hash":
        artifacts[1]["sha256"] = "invalid"
    elif damage == "authority":
        artifacts[1]["projection_authority"] = "application-owned"
    else:
        artifacts.append(deepcopy(artifacts[1]))
    part = ToolResultPart(tool_call_id="call", tool_name="tool", **payload["result"])
    message = Message.tool_result(results=[part])
    if damage == "duplicate":
        with pytest.raises(AssertionError, match="exactly one artifact reference"):
            redact_runtime_message_for_boundary(
                message, redactor=SecretRedactor(), field_name="transcript"
            )
        return
    assert read_runtime_tool_result_message(part.model_dump(mode="json")) is None
    projected = redact_runtime_message_for_boundary(
        message, redactor=SecretRedactor("private-preview"), field_name="transcript"
    )
    assert projected.content[0].content == REDACTED_SECRET


@pytest.mark.parametrize("status", ["unchanged", "failed"])
@pytest.mark.parametrize("characters", ["x", "xé", "x界", "x🙂"])
@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
def test_inline_projection_bounds_redaction_expansion_at_publication(status, characters, project):
    payload = projection_payload(status, characters * 10000)
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    output = project(event, redactor=SecretRedactor("x"), **kwargs)
    result = output.payload["result"]
    record = output.payload["tool_result_projection"]
    assert len(result["content"].encode("utf-8")) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert result["content"].endswith("[tool result truncated after redaction]")
    assert "x" not in result["content"]
    assert record["projected_bytes"] == len(result["content"].encode("utf-8"))
    assert record["projected_token_estimate"] == (len(result["content"]) + 3) // 4
    assert record["status"] == status
    assert {key: result[key] for key in ("artifacts", "structured", "is_error")} == {
        key: payload["result"][key] for key in ("artifacts", "structured", "is_error")
    }
    # Persisted reconstruction must not need the process-local prepared value.
    restored = Event.model_validate(output.model_dump(mode="json"))
    repeated = project_persisted_runtime_event(restored, sequence=2, redactor=SecretRedactor("x"))
    assert repeated.payload["result"] == result
    assert repeated.payload["tool_result_projection"] == record


@pytest.mark.parametrize("status", ["unchanged", "failed"])
@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
def test_inline_projection_truncation_preserves_redaction_markers_on_replay(status, project):
    payload = projection_payload(status, "x" * 10000)
    redactor = SecretRedactor(["x", "R", "redaction"])
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    output = project(event, redactor=redactor, **kwargs)
    result = output.payload["result"]
    record = output.payload["tool_result_projection"]
    content = result["content"]
    assert len(content.encode("utf-8")) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert redactor.redact_text(content) == content
    assert record["projected_bytes"] == len(content.encode("utf-8"))
    assert record["projected_token_estimate"] == (len(content) + 3) // 4
    # Rebuild from durable data each time; no process-local authority is needed.
    for sequence in (2, 3):
        restored = Event.model_validate(output.model_dump(mode="json"))
        output = project_persisted_runtime_event(restored, sequence=sequence, redactor=redactor)
        assert output.payload["result"] == result
        assert output.payload["tool_result_projection"] == record


@pytest.mark.parametrize("status", ["unchanged", "failed"])
def test_inline_projection_bounds_content_after_later_terminal_policy(monkeypatch, status):
    from cayu.runtime import _event_projection as events

    payload = projection_payload(status)
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    restore = events.restore_attested_event_result

    def replace_after_preparation(event, **kwargs):
        restore(event, **kwargs)
        kwargs["redacted_payload"]["result"]["content"] = "界" * 30000

    monkeypatch.setattr(events, "restore_attested_event_result", replace_after_preparation)
    output = prepare_new_runtime_event(event, redactor=SecretRedactor())
    content = output.payload["result"]["content"]
    assert len(content.encode("utf-8")) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert content.endswith("[tool result truncated after redaction]")
    assert output.payload["tool_result_projection"]["projected_bytes"] == len(content.encode())


def test_unprojected_results_retain_default_off_content_semantics():
    result = ToolResult(content="界" * 30000)
    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="session",
        payload={"result": result.model_dump(mode="json")},
    )
    assert prepare_new_runtime_event(event, redactor=SecretRedactor()).payload["result"] == (
        result.model_dump(mode="json")
    )
    message = Message.tool_result(
        results=[ToolResultPart(tool_call_id="call", tool_name="tool", content=result.content)]
    )
    assert (
        redact_runtime_message_for_boundary(
            message, redactor=SecretRedactor(), field_name="transcript"
        )
        == message
    )


def test_inline_truncation_cannot_reconstruct_a_secret_at_the_notice_boundary():
    payload = projection_payload("unchanged")
    source = read_runtime_tool_result_projection(payload)
    redactor = SecretRedactor("z\n[tool")
    prepared = PreparedToolResultProjection(
        source, redacted_result=payload["result"], redactor=redactor
    )
    notice = "\n[tool result truncated after redaction]"
    prefix_size = MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES - len(notice.encode())
    final_content = "a" * (prefix_size - 1) + "z" + "tail" * 100
    publication = prepared.finish(final_content)
    assert "z\n[tool" not in publication.content
    assert len(publication.content.encode()) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert publication.record_payload()["projected_bytes"] == len(publication.content.encode())


@pytest.mark.parametrize("status", ["unchanged", "failed"])
@pytest.mark.parametrize(
    "malformed_control", ["effect", "workspace_mutation_capture_status", "registration_state"]
)
@pytest.mark.parametrize("project", [project_runtime_event, project_persisted_runtime_event])
def test_malformed_replay_bounds_valid_inline_content(status, malformed_control, project):
    payload = projection_payload(status, "x" * 10000)
    payload[malformed_control] = "bogus"
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    redactor = SecretRedactor("x")
    output = project(event, sequence=1, redactor=redactor)
    content = output.payload["result"]["content"]
    assert len(content.encode()) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert redactor.redact_text(content) == content
    # Bounding content must not re-admit evidence discarded with bad controls.
    assert output.payload["tool_result_projection"] == payload["tool_result_projection"]
    if malformed_control == "workspace_mutation_capture_status":
        assert output.payload[malformed_control] == "bogus"
    else:
        assert malformed_control not in output.payload
    with pytest.raises(ValueError):
        prepare_new_runtime_event(event, redactor=redactor)


@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
def test_externalized_final_bound_retains_reference_after_later_policy(monkeypatch, project):
    from cayu.runtime import _event_projection as events

    payload = projection_payload()
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    restore = events.restore_attested_event_result

    def replace_after_preparation(event, **kwargs):
        restore(event, **kwargs)
        kwargs["redacted_payload"]["result"]["content"] = "界" * 30000

    with monkeypatch.context() as patch:
        patch.setattr(events, "restore_attested_event_result", replace_after_preparation)
        kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
        output = project(event, redactor=SecretRedactor(), **kwargs)
    result = output.payload["result"]
    assert len(result["content"].encode()) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert result["content"].count("[cayu tool-result projection reference]") == 1
    assert payload["tool_result_projection"]["artifact_id"] in result["content"]
    assert read_runtime_tool_result_projection(dict(output.payload)) is not None
    repeated = project_persisted_runtime_event(
        Event.model_validate(output.model_dump(mode="json")), sequence=2, redactor=SecretRedactor()
    )
    assert repeated.payload["result"] == result
    assert repeated.payload["tool_result_projection"] == output.payload["tool_result_projection"]


@pytest.mark.parametrize(
    "project", [prepare_new_runtime_event, project_runtime_event, project_persisted_runtime_event]
)
def test_externalized_preview_retains_complete_markers_across_event_and_message_replay(project):
    payload = projection_payload(content="x" * 10000)
    redactor = SecretRedactor(["x", "R"])
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    kwargs = {} if project is prepare_new_runtime_event else {"sequence": 1}
    output = project(event, redactor=redactor, **kwargs)
    message = Message.tool_result(
        results=[ToolResultPart(tool_call_id="call", tool_name="tool", **payload["result"])]
    )
    message = redact_runtime_message_for_boundary(
        message, redactor=redactor, field_name="transcript"
    )
    result = output.payload["result"]
    assert message.content[0].content == result["content"]
    prefix = result["content"].split("\n[cayu tool-result projection reference]\n")[0]
    assert prefix == REDACTED_SECRET * (len(prefix) // len(REDACTED_SECRET))
    for sequence in (2, 3):
        output = project_persisted_runtime_event(
            Event.model_validate(output.model_dump(mode="json")),
            sequence=sequence,
            redactor=redactor,
        )
        message = redact_runtime_message_for_boundary(
            Message.model_validate(message.model_dump(mode="json")),
            redactor=redactor,
            field_name="transcript",
        )
        assert output.payload["result"] == result
        assert message.content[0].content == result["content"]
        assert output.payload["tool_result_projection"]["projected_bytes"] == len(
            result["content"].encode()
        )


@pytest.mark.parametrize("status", ["unchanged", "failed"])
def test_rotated_registry_redacts_truncation_notice_and_replays_stably(status):
    payload = projection_payload(status, "x" * 10000)
    event = event_with_runtime_nested_payload_authority(
        Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", payload=payload),
        ("tool_result_projection", "policy_id"),
    )
    first = prepare_new_runtime_event(event, redactor=SecretRedactor("x"))
    redactor = SecretRedactor(["x", "tool", "result"])
    final = prepare_new_runtime_event(first, redactor=redactor)
    content = final.payload["result"]["content"]
    assert content != first.payload["result"]["content"]
    assert "tool" not in content and "result" not in content
    assert redactor.redact_text(content) == content
    assert len(content.encode()) <= MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
    assert final.payload["tool_result_projection"]["projected_bytes"] == len(content.encode())
    repeated = project_persisted_runtime_event(
        Event.model_validate(final.model_dump(mode="json")), sequence=2, redactor=redactor
    )
    assert repeated.payload == final.payload


def test_inline_transcript_without_projection_evidence_keeps_expanded_content():
    message = Message.tool_result(
        results=[ToolResultPart(tool_call_id="call", tool_name="tool", content="x" * 10000)]
    )
    projected = redact_runtime_message_for_boundary(
        message, redactor=SecretRedactor("x"), field_name="transcript"
    )
    assert projected.content[0].content == REDACTED_SECRET * 10000
    assert len(projected.content[0].content.encode()) > MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
