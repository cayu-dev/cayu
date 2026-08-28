from __future__ import annotations

import asyncio
import json
import os
import shutil
from base64 import standard_b64encode
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pytest

from cayu import FileAttachment, FileAttachmentKind, Message, ResolvedFileAttachment
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.core.messages import FilePart, MessageRole, TextPart
from cayu.evals.external import (
    EXTERNAL_PROCESS_PROTOCOL_VERSION,
    ExternalBodyReleaseV1,
    ExternalProcessTargetIdentityV1,
    ExternalTrialEnvelopeV1,
    ExternalTrialIdentityV1,
    external_body_content_revision,
)
from cayu.evals.external_container import (
    _DOCKER_CLI_MAX_CAPTURE_BYTES,
    EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION,
    EXTERNAL_CONTAINER_RUNNER_REVISION,
    EXTERNAL_CONTAINER_STREAM_PROTOCOL,
    ExternalContainerOperationAdapter,
    _DockerCliClient,
    _DockerResult,
    external_container_environment_revision,
)
from cayu.providers import ModelRequest
from cayu.providers.operations import (
    ProviderOperationStartRecoveryRequest,
    ProviderOperationStartRequest,
    ProviderOperationStatus,
)

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64
_D = "sha256:" + "d" * 64
_E = "sha256:" + "e" * 64


class _FakeDocker:
    def __init__(self, *, complete_on_start: bool = True, output: bytes | None = None) -> None:
        self.complete_on_start = complete_on_start
        self.output = (
            output
            or json.dumps(
                {
                    "schema_version": 1,
                    "output": "Approved",
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                },
                separators=(",", ":"),
            ).encode()
        )
        self.calls: list[tuple[str, ...]] = []
        self.states: dict[str, dict[str, object]] = {}
        self.labels: dict[str, dict[str, str]] = {}

    async def run(self, args: Sequence[str]) -> _DockerResult:
        call = tuple(args)
        self.calls.append(call)
        if args[0] == "create":
            name = args[args.index("--name") + 1]
            self.states[name] = {"Status": "created", "Running": False, "ExitCode": 0}
            self.labels[name] = {
                args[index + 1].split("=", 1)[0]: args[index + 1].split("=", 1)[1]
                for index, item in enumerate(args)
                if item == "--label"
            }
            return _DockerResult(0, name.encode())
        if args[0] == "cp":
            return _DockerResult(0)
        if args[0] == "start":
            name = args[-1]
            self.states[name] = (
                {"Status": "exited", "Running": False, "ExitCode": 0, "OOMKilled": False}
                if self.complete_on_start
                else {"Status": "running", "Running": True, "ExitCode": 0}
            )
            return _DockerResult(0, name.encode())
        if args[0] == "inspect":
            name = args[-1]
            state = self.states.get(name)
            if state is None:
                return _DockerResult(1, stderr=b"not found")
            if args[2] == "{{json .Config.Labels}}":
                return _DockerResult(0, json.dumps(self.labels[name]).encode())
            return _DockerResult(0, json.dumps(state).encode())
        if args[0] == "logs":
            return _DockerResult(0, self.output)
        if args[0] == "stop":
            name = args[-1]
            self.states[name] = {
                "Status": "exited",
                "Running": False,
                "ExitCode": 143,
                "OOMKilled": False,
            }
            return _DockerResult(0, name.encode())
        if args[0] == "rm":
            name = args[-1]
            self.states.pop(name, None)
            self.labels.pop(name, None)
            return _DockerResult(0, name.encode())
        raise AssertionError(f"unexpected Docker call: {args!r}")


def _body(root: Path) -> ExternalBodyReleaseV1:
    root.mkdir()
    (root / "private_runtime.py").write_text("# pinned private runtime\n", encoding="utf-8")
    (root / "agent.py").write_text("# mutable candidate anatomy\n", encoding="utf-8")
    return ExternalBodyReleaseV1.from_directory(
        root,
        private_runtime_path="private_runtime.py",
        launch_protocol_revision=_A,
        entrypoint=("{body}/private_runtime.py", "{body}/agent.py", "{request}"),
    )


def _identity(root: Path) -> ExternalProcessTargetIdentityV1:
    image = "example.invalid/candidate@sha256:" + "f" * 64
    return ExternalProcessTargetIdentityV1.create(
        body=_body(root),
        evaluator_runtime_revision=_B,
        target_implementation_revision=_C,
        runner_revision=EXTERNAL_CONTAINER_RUNNER_REVISION,
        environment_revision=external_container_environment_revision(
            image=image,
            runtime="runsc",
        ),
        reset_contract_revision=EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION,
        evidence_policy_revision=_B,
    )


def _request(
    identity: ExternalProcessTargetIdentityV1,
    trial_number: int = 1,
    *,
    attachment_content: bytes | None = None,
    message_text: str = "Approve this.",
) -> ModelRequest:
    trial = ExternalTrialIdentityV1.create(
        native_run_id="native-run-one",
        target_key="external-agent",
        target_revision=identity.revision,
        corpus_revision=_A,
        suite_id="external-suite",
        suite_revision=_B,
        case_id="external-case",
        case_revision=_C,
        trial_number=trial_number,
    )
    messages = [
        ExternalTrialEnvelopeV1(trial=trial).message(),
        Message.text("user", message_text),
    ]
    options = {}
    if attachment_content is not None:
        attachment = FileAttachment(
            artifact_id="fixture-preservation-document",
            kind=FileAttachmentKind.DOCUMENT,
            filename="preservation.pdf",
            content_type="application/pdf",
            size_bytes=len(attachment_content),
        )
        messages[-1] = Message(
            role=MessageRole.USER,
            content=(
                TextPart(text="Approve this attachment."),
                FilePart(attachment=attachment.model_dump(mode="json")),
            ),
        )
        digest = sha256(attachment_content).hexdigest()
        resolved = ResolvedFileAttachment(
            artifact_id=attachment.artifact_id,
            kind=attachment.kind,
            filename=attachment.filename,
            content_type=attachment.content_type,
            data_base64=standard_b64encode(attachment_content).decode("ascii"),
            content_sha256=digest,
        )
        options = {
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                attachment.artifact_id: resolved.model_dump(mode="json", exclude_none=True)
            }
        }
    return ModelRequest(
        model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
        messages=messages,
        options=options,
    )


def _adapter(
    tmp_path: Path,
    identity: ExternalProcessTargetIdentityV1,
    fake: _FakeDocker,
) -> ExternalContainerOperationAdapter:
    return ExternalContainerOperationAdapter(
        identity=identity,
        body_root=tmp_path / "body",
        state_root=tmp_path / "state",
        image="example.invalid/candidate@sha256:" + "f" * 64,
        runtime="runsc",
        docker_path="/usr/bin/docker",
        poll_seconds=0.01,
        _client=fake,
    )


async def _events(connection) -> list:
    return [event async for event in connection.events]


@pytest.mark.process
@pytest.mark.anyio
async def test_docker_cli_capture_kills_unbounded_output() -> None:
    executable = shutil.which("yes")
    if executable is None:
        pytest.skip("yes executable is unavailable")

    result = await asyncio.wait_for(_DockerCliClient(executable).run(()), timeout=5)

    assert result.exit_code != 0
    assert len(result.stdout) == _DOCKER_CLI_MAX_CAPTURE_BYTES + 1
    assert result.stderr == b""


@pytest.mark.anyio
async def test_hardened_container_start_is_fresh_bounded_and_reconnectable(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    adapter = _adapter(tmp_path, identity, fake)

    connection = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity, attachment_content=b"%PDF-1.7 fixture\n"),
            idempotency_key="stable-operation-key",
        )
    )
    events = await _events(connection)

    assert connection.status is ProviderOperationStatus.COMPLETED
    assert connection.state.stream_protocol == EXTERNAL_CONTAINER_STREAM_PROTOCOL
    assert [event.type for event in events] == ["text_delta", "completed"]
    create = next(call for call in fake.calls if call[0] == "create")
    assert create[create.index("--runtime") :][:2] == ("--runtime", "runsc")
    assert create[create.index("--network") :][:2] == ("--network", "none")
    assert "--read-only" in create
    assert create[create.index("--cap-drop") :][:2] == ("--cap-drop", "ALL")
    assert create[create.index("--security-opt") :][:2] == ("--security-opt", "no-new-privileges")
    assert "--mount" not in create
    assert "65532:65532" in create
    assert "/cayu/body/private_runtime.py" in create
    assert "/cayu/request.json" in create
    operation_dir = tmp_path / "state" / connection.state.operation_id
    authority = json.loads((operation_dir / "authority.json").read_text())
    trial_revision = connection.state.recovery_metadata.opaque["trial_revision"]
    assert type(trial_revision) is str
    launch = json.loads((operation_dir / "payload" / "request.json").read_text())
    launch_request_sha256 = sha256(
        (operation_dir / "payload" / "request.json").read_bytes()
    ).hexdigest()
    assert authority == {
        "environment_revision": identity.environment_revision,
        "idempotency_sha256": sha256(trial_revision.encode("utf-8")).hexdigest(),
        "launch_request_sha256": launch_request_sha256,
        "schema_version": 2,
        "target_revision": identity.revision,
        "trial_revision": trial_revision,
    }
    assert connection.state.recovery_metadata.opaque == {
        key: authority[key] for key in authority if key != "schema_version"
    }
    assert external_body_content_revision(operation_dir / "payload" / "body") == (
        identity.body.content_revision
    )
    assert (
        launch["envelope"]["trial"]["revision"]
        == (connection.state.recovery_metadata.opaque["trial_revision"])
    )
    assert launch["request"]["messages"][0]["content"][0]["text"] == ("Approve this attachment.")
    attached = launch["request"]["options"][RESOLVED_FILE_ATTACHMENTS_OPTION][
        "fixture-preservation-document"
    ]
    assert attached["content_sha256"] == sha256(b"%PDF-1.7 fixture\n").hexdigest()
    assert attached["data_base64"] == standard_b64encode(b"%PDF-1.7 fixture\n").decode("ascii")

    recovered_adapter = _adapter(tmp_path, identity, fake)
    recovered = await recovered_adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="stable-operation-key")
    )
    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert sum(call[0] == "create" for call in fake.calls) == 1
    assert sum(call[0] == "rm" for call in fake.calls) == 1
    assert connection.state.operation_id not in fake.states
    terminal = json.loads((operation_dir / "terminal.json").read_text())
    assert terminal["schema_version"] == 2
    assert terminal["launch_request_sha256"] == launch_request_sha256
    assert terminal["trial_revision"] == trial_revision


@pytest.mark.anyio
async def test_trial_identity_deduplicates_redispatch_with_new_runtime_key(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    adapter = _adapter(tmp_path, identity, fake)

    first = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity), idempotency_key="first-runtime-start-key"
        )
    )
    redispatched = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity), idempotency_key="replacement-runtime-start-key"
        )
    )

    assert redispatched.status is ProviderOperationStatus.COMPLETED
    assert redispatched.state.operation_id == first.state.operation_id
    assert sum(call[0] == "create" for call in fake.calls) == 1
    assert sum(call[0] == "start" for call in fake.calls) == 1
    recovered = await adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="replacement-runtime-start-key")
    )
    assert recovered.state.operation_id == first.state.operation_id


@pytest.mark.anyio
async def test_trial_identity_rejects_a_different_resolved_launch_request(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    adapter = _adapter(tmp_path, identity, fake)
    first = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity), idempotency_key="first-runtime-start-key"
        )
    )

    with pytest.raises(ValueError, match="launch request changed identity"):
        await adapter.start(
            ProviderOperationStartRequest(
                request=_request(identity, message_text="A different resolved request."),
                idempotency_key="replacement-runtime-start-key",
            )
        )

    assert sum(call[0] == "create" for call in fake.calls) == 1
    retained = json.loads(
        (tmp_path / "state" / first.state.operation_id / "payload" / "request.json").read_text()
    )
    assert retained["request"]["messages"][0]["content"][0]["text"] == "Approve this."


@pytest.mark.anyio
async def test_candidate_reported_usage_is_only_untrusted_diagnostic_output(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(
        output=json.dumps(
            {
                "schema_version": 1,
                "output": "Approved",
                "usage": {"input_tokens": 9001, "output_tokens": 7, "total_tokens": 9008},
            },
            separators=(",", ":"),
        ).encode()
    )
    adapter = _adapter(tmp_path, identity, fake)

    connection = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity), idempotency_key="untrusted-usage-key"
        )
    )
    events = await _events(connection)
    completed = events[-1]

    assert completed.type == "completed"
    assert "usage" not in completed.payload
    assert "usage_metrics" not in completed.payload
    assert completed.payload["external_candidate_diagnostics"] == {
        "usage_trust": "candidate_reported_untrusted",
        "reported_usage": {
            "input_tokens": 9001,
            "output_tokens": 7,
            "total_tokens": 9008,
        },
    }
    assert completed.completion is not None


@pytest.mark.anyio
async def test_terminal_receipt_replay_rejects_another_trials_receipt(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    adapter = _adapter(tmp_path, identity, fake)
    first = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity, trial_number=1), idempotency_key="first-terminal-key"
        )
    )
    second = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity, trial_number=2), idempotency_key="second-terminal-key"
        )
    )
    first_terminal = tmp_path / "state" / first.state.operation_id / "terminal.json"
    second_terminal = tmp_path / "state" / second.state.operation_id / "terminal.json"
    second_terminal.write_bytes(first_terminal.read_bytes())

    replayed = await adapter.retrieve(second.state)

    assert replayed.status is ProviderOperationStatus.UNAVAILABLE
    assert replayed.events[0].payload["provider_error_code"] == (
        "external_container_identity_mismatch"
    )


@pytest.mark.anyio
async def test_start_alias_is_not_visible_before_recoverable_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    adapter = _adapter(tmp_path, identity, fake)
    bind_start_alias = adapter._bind_start_alias

    def cancel_after_alias(*args, **kwargs) -> None:
        bind_start_alias(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(adapter, "_bind_start_alias", cancel_after_alias)
    with pytest.raises(asyncio.CancelledError):
        await adapter.start(
            ProviderOperationStartRequest(
                request=_request(identity), idempotency_key="cancelled-after-alias"
            )
        )

    operation_dirs = [
        path for path in (tmp_path / "state").iterdir() if path.name != "start-aliases"
    ]
    assert len(operation_dirs) == 1
    assert json.loads((operation_dirs[0] / "phase.json").read_text()) == {
        "schema_version": 1,
        "phase": "preparing",
    }
    assert sum(call[0] == "create" for call in fake.calls) == 0

    recovered = await _adapter(tmp_path, identity, fake).recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="cancelled-after-alias")
    )

    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert sum(call[0] == "create" for call in fake.calls) == 1


@pytest.mark.anyio
async def test_container_recovery_fails_closed_on_identity_label_drift(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="drift-key")
    )
    fake.labels[connection.state.operation_id]["dev.cayu.eval.trial"] = _E

    snapshot = await adapter.retrieve(connection.state)

    assert snapshot.status is ProviderOperationStatus.UNAVAILABLE
    assert snapshot.events[0].payload["provider_error_code"] == (
        "external_container_identity_mismatch"
    )


@pytest.mark.anyio
async def test_container_created_phase_reconciles_completed_effect_without_restart(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="partial-key")
    )
    operation_dir = tmp_path / "state" / connection.state.operation_id
    (operation_dir / "phase.json").write_text(
        '{"phase":"created","schema_version":1}', encoding="utf-8"
    )
    fake.states[connection.state.operation_id] = {
        "Status": "exited",
        "Running": False,
        "ExitCode": 0,
        "OOMKilled": False,
    }
    starts_before = sum(call[0] == "start" for call in fake.calls)

    recovered = await adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="partial-key")
    )
    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert sum(call[0] == "start" for call in fake.calls) == starts_before


@pytest.mark.anyio
async def test_container_prepared_phase_recovers_before_external_create(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="prepared-key")
    )
    operation_dir = tmp_path / "state" / connection.state.operation_id
    (operation_dir / "phase.json").write_text(
        '{"phase":"prepared","schema_version":1}', encoding="utf-8"
    )
    fake.calls.clear()
    fake.states.clear()
    fake.labels.clear()
    fake.complete_on_start = True

    recovered = await adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="prepared-key")
    )

    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert sum(call[0] == "create" for call in fake.calls) == 1
    assert sum(call[0] == "start" for call in fake.calls) == 1


@pytest.mark.anyio
async def test_container_preparing_phase_recovers_partial_body_before_effect(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="preparing-key")
    )
    operation_dir = tmp_path / "state" / connection.state.operation_id
    snapshot = operation_dir / "payload" / "body"
    snapshot.chmod(0o700)
    candidate = snapshot / "agent.py"
    candidate.chmod(0o600)
    candidate.write_text("partial", encoding="utf-8")
    (operation_dir / "phase.json").write_text(
        '{"phase":"preparing","schema_version":1}', encoding="utf-8"
    )
    fake.calls.clear()
    fake.states.clear()
    fake.labels.clear()
    fake.complete_on_start = True

    recovered = await adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="preparing-key")
    )

    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert external_body_content_revision(snapshot) == identity.body.content_revision
    assert sum(call[0] == "create" for call in fake.calls) == 1
    assert sum(call[0] == "start" for call in fake.calls) == 1


@pytest.mark.anyio
async def test_container_complete_copy_resumes_one_exact_start(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="copied-key")
    )
    operation_dir = tmp_path / "state" / connection.state.operation_id
    (operation_dir / "phase.json").write_text(
        '{"phase":"copied","schema_version":1}', encoding="utf-8"
    )
    fake.states[connection.state.operation_id] = {
        "Status": "created",
        "Running": False,
        "ExitCode": 0,
    }
    fake.complete_on_start = True
    starts_before = sum(call[0] == "start" for call in fake.calls)

    recovered = await adapter.recover_start(
        ProviderOperationStartRecoveryRequest(idempotency_key="copied-key")
    )

    assert recovered.status is ProviderOperationStatus.COMPLETED
    assert sum(call[0] == "start" for call in fake.calls) == starts_before + 1


@pytest.mark.anyio
async def test_container_cancel_targets_exact_live_operation(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="cancel-key")
    )
    trial_label = fake.labels[connection.state.operation_id]["dev.cayu.eval.trial"]

    cancelled = await adapter.cancel(connection.state)

    assert connection.status is ProviderOperationStatus.IN_PROGRESS
    assert cancelled.status is ProviderOperationStatus.CANCELLED
    stopped = next(call for call in fake.calls if call[:3] == ("stop", "--time", "5"))
    assert stopped[-1] == connection.state.operation_id
    assert trial_label == connection.state.recovery_metadata.opaque["trial_revision"]
    assert sum(call[0] == "create" for call in fake.calls) == 1
    assert sum(call[0] == "start" for call in fake.calls) == 1


@pytest.mark.anyio
async def test_container_marks_invalid_output_incomplete_without_retry(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(output=b"not-json")
    adapter = _adapter(tmp_path, identity, fake)

    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="bad-output")
    )
    events = await _events(connection)

    assert connection.status is ProviderOperationStatus.FAILED
    assert len(events) == 1
    assert events[0].payload["provider_error_code"] == "external_container_incomplete"
    assert events[0].payload["retryable"] is False


@pytest.mark.anyio
async def test_container_preserves_oom_and_unknown_dispositions(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker(complete_on_start=False)
    adapter = _adapter(tmp_path, identity, fake)
    connection = await adapter.start(
        ProviderOperationStartRequest(request=_request(identity), idempotency_key="oom-key")
    )
    name = connection.state.operation_id
    fake.states[name] = {
        "Status": "exited",
        "Running": False,
        "ExitCode": 137,
        "OOMKilled": True,
    }

    oom = await adapter.retrieve(connection.state)

    assert oom.status is ProviderOperationStatus.FAILED
    assert oom.events[0].payload["provider_error_code"] == "external_container_oom_killed"
    unknown_connection = await adapter.start(
        ProviderOperationStartRequest(
            request=_request(identity, trial_number=2), idempotency_key="unknown-key"
        )
    )
    fake.states.clear()

    unknown = await adapter.retrieve(unknown_connection.state)

    assert unknown.status is ProviderOperationStatus.UNAVAILABLE
    assert unknown.events[0].payload["provider_error_code"] == "external_container_unknown"


def test_container_rejects_body_drift_and_soft_isolation(tmp_path: Path) -> None:
    identity = _identity(tmp_path / "body")
    fake = _FakeDocker()
    _adapter(tmp_path, identity, fake)
    mismatched_body = ExternalBodyReleaseV1.create(
        content_revision=identity.body.content_revision,
        private_runtime_path=identity.body.private_runtime_path,
        private_runtime_revision=_E,
        launch_protocol_revision=identity.body.launch_protocol_revision,
        entrypoint=identity.body.entrypoint,
    )
    mismatched_identity = ExternalProcessTargetIdentityV1.create(
        body=mismatched_body,
        evaluator_runtime_revision=identity.evaluator_runtime_revision,
        target_implementation_revision=identity.target_implementation_revision,
        runner_revision=identity.runner_revision,
        environment_revision=identity.environment_revision,
        reset_contract_revision=identity.reset_contract_revision,
        evidence_policy_revision=identity.evidence_policy_revision,
    )
    with pytest.raises(ValueError, match="private runtime identity"):
        _adapter(tmp_path, mismatched_identity, fake)
    with pytest.raises(ValueError, match="runsc or Kata"):
        ExternalContainerOperationAdapter(
            identity=identity,
            body_root=tmp_path / "body",
            state_root=tmp_path / "other-state",
            image="example.invalid/candidate@sha256:" + "f" * 64,
            runtime="runc",
            docker_path="/usr/bin/docker",
            _client=fake,
        )
    (tmp_path / "body" / "agent.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="body identity"):
        _adapter(tmp_path, identity, fake)


@pytest.mark.process
@pytest.mark.anyio
async def test_real_hardened_containers_run_multifile_body_with_fresh_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = os.environ.get("CAYU_TEST_EXTERNAL_CONTAINER_IMAGE")
    if image is None:
        pytest.skip("CAYU_TEST_EXTERNAL_CONTAINER_IMAGE is not configured")
    assert image is not None
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    runtime = os.environ.get("CAYU_TEST_EXTERNAL_CONTAINER_RUNTIME", "runsc")
    body_root = Path(__file__).parent / "fixtures" / "external_agent"
    body = ExternalBodyReleaseV1.from_directory(
        body_root,
        private_runtime_path="private_cayu_runtime.py",
        launch_protocol_revision=_A,
        entrypoint=(
            "python3",
            "{body}/private_cayu_runtime.py",
            "{body}/agent.py",
            "{request}",
        ),
    )
    identity = ExternalProcessTargetIdentityV1.create(
        body=body,
        evaluator_runtime_revision=_B,
        target_implementation_revision=_C,
        runner_revision=EXTERNAL_CONTAINER_RUNNER_REVISION,
        environment_revision=external_container_environment_revision(
            image=image,
            runtime=runtime,
        ),
        reset_contract_revision=EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION,
        evidence_policy_revision=_D,
    )
    adapter = ExternalContainerOperationAdapter(
        identity=identity,
        body_root=body_root,
        state_root=tmp_path / "state",
        image=image,
        runtime=runtime,
        docker_path=docker,
        poll_seconds=0.05,
    )
    monkeypatch.setenv("CAYU_EVAL_TRUTH_SENTINEL", "must-not-cross-container-boundary")
    connections = []
    try:
        for trial_number in (1, 2):
            connection = await adapter.start(
                ProviderOperationStartRequest(
                    request=_request(
                        identity,
                        trial_number=trial_number,
                        attachment_content=b"%PDF-1.7 preservation fixture\n",
                    ),
                    idempotency_key=f"real-container-{trial_number}",
                )
            )
            events = await _events(connection)
            assert connection.status is ProviderOperationStatus.COMPLETED
            assert events[0].delta == "Approved"
            connections.append(connection)
        assert connections[0].state.operation_id != connections[1].state.operation_id
    finally:
        for connection in connections:
            await adapter._client.run(["rm", "-f", connection.state.operation_id])
