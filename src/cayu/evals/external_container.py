"""Hardened Docker reference adapter for external eval targets."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.evals.external import (
    EXTERNAL_PROCESS_PROTOCOL_VERSION,
    ExternalProcessTargetIdentityV1,
    ExternalTrialEnvelopeV1,
    external_body_content_revision,
    external_body_file_revision,
    external_trial_envelope_from_request,
)
from cayu.providers import ModelProviderError, ModelStreamEvent
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRecoveryRequest,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runners._docker_cli import docker_cli_env

# Runtime-resolved attachments may occupy 32 MiB before their bounded base64
# provider projection. Keep enough room for that exact material plus the rest
# of the sealed launch request without creating an unbounded transport.
EXTERNAL_CONTAINER_MAX_INPUT_BYTES = 48 << 20
EXTERNAL_CONTAINER_MAX_OUTPUT_BYTES = 1 << 20
_DOCKER_CLI_MAX_CAPTURE_BYTES = 2 << 20
EXTERNAL_CONTAINER_STREAM_PROTOCOL = "cayu.external-container.v2"
EXTERNAL_CONTAINER_RUNNER_REVISION = (
    "sha256:" + sha256(b"cayu.external-container.runner.v2").hexdigest()
)
EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION = (
    "sha256:"
    + sha256(b"fresh-container:no-network:no-mounts:read-only-root:tmpfs-workspace:v1").hexdigest()
)
_CONTAINER_NAME_PREFIX = "cayu-eval-"
_CONTAINER_LABEL_PREFIX = "dev.cayu.eval"


class _ExternalContainerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ExternalContainerUsageV1(_ExternalContainerModel):
    """Untrusted diagnostic usage reported by the candidate-private runtime."""

    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)

    @field_validator("total_tokens")
    @classmethod
    def validate_total_tokens(cls, value: int, info) -> int:
        if (
            "input_tokens" in info.data
            and "output_tokens" in info.data
            and value != info.data["input_tokens"] + info.data["output_tokens"]
        ):
            raise ValueError("total_tokens must equal input_tokens plus output_tokens.")
        return value


class ExternalContainerOutputV1(_ExternalContainerModel):
    """Bounded candidate output returned through daemon-retained container logs."""

    schema_version: Literal[1] = 1
    output: StrictStr = Field(max_length=EXTERNAL_CONTAINER_MAX_OUTPUT_BYTES)
    usage: ExternalContainerUsageV1


class ExternalContainerLaunchRequestV1(_ExternalContainerModel):
    """Authority-free input copied into a fresh container before it starts."""

    schema_version: Literal[1] = 1
    protocol: Literal["cayu.external-process.v1"] = EXTERNAL_PROCESS_PROTOCOL_VERSION
    envelope: ExternalTrialEnvelopeV1
    request: dict[str, object]


@dataclass(frozen=True, slots=True)
class _DockerResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""


class _DockerClient(Protocol):
    async def run(self, args: Sequence[str]) -> _DockerResult: ...


class _DockerCliClient:
    def __init__(self, docker_path: str) -> None:
        self._docker_path = docker_path

    async def run(self, args: Sequence[str]) -> _DockerResult:
        process = await asyncio.create_subprocess_exec(
            self._docker_path,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=docker_cli_env(),
        )
        assert process.stdout is not None
        assert process.stderr is not None

        async def read_bounded(stream: asyncio.StreamReader) -> bytes:
            retained = bytearray()
            while True:
                chunk = await stream.read(64 << 10)
                if not chunk:
                    return bytes(retained)
                retained.extend(chunk)
                if len(retained) > _DOCKER_CLI_MAX_CAPTURE_BYTES:
                    with suppress(ProcessLookupError):
                        process.kill()
                    return bytes(retained[: _DOCKER_CLI_MAX_CAPTURE_BYTES + 1])

        stdout, stderr = await asyncio.gather(
            read_bounded(process.stdout),
            read_bounded(process.stderr),
        )
        return_code = await process.wait()
        return _DockerResult(return_code, stdout, stderr)


def _pinned_image(value: str) -> str:
    value = require_durable_clean_nonblank(value, "image")
    if value.startswith("-") or any(character.isspace() for character in value):
        raise ValueError("External container image must be one non-option reference.")
    marker = "@sha256:"
    if marker not in value:
        raise ValueError("External container image must be pinned by sha256 digest.")
    digest = value.rsplit(marker, 1)[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("External container image must use a lowercase sha256 digest.")
    return value


def _hardened_runtime(value: str) -> str:
    value = require_durable_clean_nonblank(value, "runtime")
    if value != "runsc" and not value.startswith("kata"):
        raise ValueError("External containers require the runsc or Kata hardened runtime.")
    return value


def _container_name(idempotency_key: str) -> str:
    key = require_durable_clean_nonblank(idempotency_key, "idempotency_key")
    return _CONTAINER_NAME_PREFIX + sha256(key.encode("utf-8")).hexdigest()[:48]


def external_container_environment_revision(
    *,
    image: str,
    runtime: str,
    memory: str = "1g",
    cpus: str = "1.0",
    pids_limit: int = 256,
) -> str:
    """Return the exact public identity of the hardened execution environment."""

    material = {
        "schema_version": 1,
        "image": _pinned_image(image),
        "runtime": _hardened_runtime(runtime),
        "memory": require_durable_clean_nonblank(memory, "memory"),
        "cpus": require_durable_clean_nonblank(cpus, "cpus"),
        "pids_limit": pids_limit,
        "network": "none",
        "root_filesystem": "read_only",
        "capabilities": "none",
        "security_options": ["no-new-privileges"],
        "user": "65532:65532",
        "workdir": "/workspace",
        "tmpfs": [
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "/workspace:rw,nosuid,nodev,size=256m",
        ],
        "mounts": "none",
        "log_driver_limits": {"max_size": "1m", "max_files": 1},
    }
    digest = sha256(canonical_durable_json_bytes(material, "external container environment"))
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    try:
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_json_once(path: Path, value: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_json(path, maximum_bytes=2 << 20) != value:
                raise ValueError("External container terminal receipt changed outcome.") from None
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_json(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError("External container state is missing or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("External container state must be one JSON object.")
    return value


def _copy_body_snapshot(source: Path, destination: Path, expected_revision: str) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("External body reconstruction path changed identity.")
        destination.chmod(0o700)
    else:
        destination.mkdir(mode=0o700)
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("External body snapshots cannot contain symbolic links.")
        target = destination / path.relative_to(source)
        if path.is_dir():
            if target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise ValueError("External body reconstruction path changed identity.")
                target.chmod(0o700)
            else:
                target.mkdir(mode=0o700)
            continue
        if not path.is_file():
            raise ValueError("External body snapshots can contain only regular files.")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError("External body reconstruction path changed identity.")
            target.chmod(0o600)
        shutil.copyfile(path, target, follow_symlinks=False)
        target.chmod(path.stat().st_mode & 0o777)
    if external_body_content_revision(destination) != expected_revision:
        raise ValueError("Reconstructed external body does not match its selected identity.")
    for file_path in (path for path in destination.rglob("*") if path.is_file()):
        file_path.chmod(0o444 | (file_path.stat().st_mode & 0o111))
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    destination.chmod(0o555)


class ExternalContainerOperationAdapter(ProviderOperationAdapter):
    """Execute immutable candidates in fresh, recoverable hardened containers.

    Every operation has a deterministic container name derived from the exact
    external trial identity. Provider start keys are durable aliases to that
    scientific effect, so a redispatch cannot create a second container for the
    same trial. The body and input are copied into a stopped container before
    launch; runtime execution then has no mounts, network, Linux capabilities,
    writable root, or host credentials. Daemon state and bounded logs are the
    external-effect reconciliation surface after restart.
    """

    def __init__(
        self,
        *,
        identity: ExternalProcessTargetIdentityV1,
        body_root: str | Path,
        state_root: str | Path,
        image: str,
        runtime: str,
        docker_path: str | None = None,
        memory: str = "1g",
        cpus: str = "1.0",
        pids_limit: int = 256,
        poll_seconds: float = 0.1,
        _client: _DockerClient | None = None,
    ) -> None:
        if type(identity) is not ExternalProcessTargetIdentityV1:
            raise TypeError("identity must be an exact ExternalProcessTargetIdentityV1.")
        body = Path(body_root)
        state = Path(state_root)
        if not body.is_absolute() or not body.is_dir() or body.is_symlink():
            raise ValueError("body_root must be an absolute non-symlink directory.")
        if not state.is_absolute():
            raise ValueError("state_root must be an absolute path.")
        if external_body_content_revision(body) != identity.body.content_revision:
            raise ValueError("body_root does not match the selected body identity.")
        if (
            external_body_file_revision(body, identity.body.private_runtime_path)
            != identity.body.private_runtime_revision
        ):
            raise ValueError("body_root does not match the selected private runtime identity.")
        if type(pids_limit) is not int or not 16 <= pids_limit <= 4096:
            raise ValueError("pids_limit must be an integer from 16 through 4096.")
        if type(poll_seconds) is not float or not 0.01 <= poll_seconds <= 1.0:
            raise ValueError("poll_seconds must be a float from 0.01 through 1.0.")
        docker = docker_path or shutil.which("docker")
        if _client is None and docker is None:
            raise ValueError("docker_path is required when the Docker CLI is unavailable.")
        if docker is not None and not Path(docker).is_absolute():
            raise ValueError("docker_path must resolve to an absolute executable path.")
        if state.is_symlink() or (state.exists() and not state.is_dir()):
            raise ValueError("state_root must be a non-symlink directory.")
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        state.chmod(0o700)
        self.identity = ExternalProcessTargetIdentityV1.model_validate(
            identity.model_dump(mode="json")
        )
        self.body_root = body
        self.state_root = state
        self._start_alias_root = state / "start-aliases"
        self._start_alias_root.mkdir(mode=0o700, exist_ok=True)
        if self._start_alias_root.is_symlink() or not self._start_alias_root.is_dir():
            raise ValueError("External container alias state changed identity.")
        self._start_alias_root.chmod(0o700)
        self.image = _pinned_image(image)
        self.runtime = _hardened_runtime(runtime)
        self.memory = require_durable_clean_nonblank(memory, "memory")
        self.cpus = require_durable_clean_nonblank(cpus, "cpus")
        self.pids_limit = pids_limit
        self.poll_seconds = poll_seconds
        expected_environment = external_container_environment_revision(
            image=self.image,
            runtime=self.runtime,
            memory=self.memory,
            cpus=self.cpus,
            pids_limit=self.pids_limit,
        )
        if self.identity.environment_revision != expected_environment:
            raise ValueError("External target environment identity does not match its container.")
        if self.identity.runner_revision != EXTERNAL_CONTAINER_RUNNER_REVISION:
            raise ValueError("External target runner identity does not match this adapter.")
        if self.identity.reset_contract_revision != EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION:
            raise ValueError("External target reset identity does not match fresh-container reset.")
        self._client = (
            _DockerCliClient(docker or "/unavailable/docker") if _client is None else _client
        )

    @property
    def start_idempotency_support(self) -> ProviderOperationStartIdempotencySupport:
        return ProviderOperationStartIdempotencySupport.EXACT

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        envelope, candidate_request = external_trial_envelope_from_request(
            request.request,
            expected_target_revision=self.identity.revision,
        )
        launch = ExternalContainerLaunchRequestV1(
            envelope=envelope,
            request=candidate_request.model_dump(mode="json", round_trip=True, warnings="none"),
        )
        request_bytes = canonical_durable_json_bytes(
            launch.model_dump(mode="json"),
            "external container launch request",
        )
        if len(request_bytes) > EXTERNAL_CONTAINER_MAX_INPUT_BYTES:
            raise ValueError("External container input exceeds its byte limit.")
        launch_request_sha256 = sha256(request_bytes).hexdigest()
        name = _container_name(envelope.trial.revision)
        operation_dir = self.state_root / name
        start_key_digest = sha256(request.idempotency_key.encode("utf-8")).hexdigest()
        effect_key_digest = sha256(envelope.trial.revision.encode("utf-8")).hexdigest()
        authority = self._authority_document(
            envelope.trial.revision,
            effect_key_digest=effect_key_digest,
            launch_request_sha256=launch_request_sha256,
        )
        argv = self._entrypoint_argv()
        if not any("{body}" in item for item in self.identity.body.entrypoint) or not any(
            "{request}" in item for item in self.identity.body.entrypoint
        ):
            raise ValueError("Container entrypoint must reference {body} and {request}.")
        try:
            operation_dir.mkdir(mode=0o700)
        except FileExistsError:
            if operation_dir.is_symlink() or not operation_dir.is_dir():
                raise ValueError("External container operation state changed identity.") from None
            _, retained_authority = self._load_bound_operation(name, operation_dir)
            if retained_authority != authority:
                raise ValueError("External container launch request changed identity.") from None
            self._bind_start_alias(
                start_key_digest,
                name=name,
                authority=authority,
            )
            return await self._recover_connection(name, operation_dir)
        effect_key_path = operation_dir / "idempotency.sha256"
        effect_key_path.write_text(effect_key_digest, encoding="ascii")
        effect_key_path.chmod(0o600)
        try:
            _write_json(
                operation_dir / "authority.json",
                authority,
            )
            payload = operation_dir / "payload"
            payload.mkdir(mode=0o700)
            (payload / "request.json").write_bytes(request_bytes)
            (payload / "request.json").chmod(0o444)
            _write_json(
                operation_dir / "phase.json",
                {"schema_version": 1, "phase": "preparing"},
            )
            self._bind_start_alias(
                start_key_digest,
                name=name,
                authority=authority,
            )
            self._prepare_payload(payload)
            _write_json(
                operation_dir / "phase.json",
                {"schema_version": 1, "phase": "prepared"},
            )
            await self._advance_admission(name, operation_dir, argv)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            _write_json(
                operation_dir / "phase.json",
                {
                    "schema_version": 1,
                    "phase": "unavailable",
                    "disposition": "unavailable",
                    "diagnostic": f"External container admission failed: {type(exc).__name__}.",
                },
            )
        return await self._connection(self._state(name, authority=authority))

    async def recover_start(
        self,
        request: ProviderOperationStartRecoveryRequest,
    ) -> ProviderOperationConnection:
        start_key_digest = sha256(request.idempotency_key.encode("utf-8")).hexdigest()
        name, alias_authority = self._start_alias(start_key_digest)
        operation_dir = self.state_root / name
        _, retained_authority = self._load_bound_operation(name, operation_dir)
        if retained_authority != alias_authority:
            raise ValueError("External container start alias changed identity.")
        return await self._recover_connection(name, operation_dir)

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        copied = self._validated_state(state)
        status, events = await self._status_and_events(copied)
        return ProviderOperationSnapshot(state=copied, status=status, events=events)

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        return await self._connection(self._validated_state(state))

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        copied = self._validated_state(state)
        status, events = await self._status_and_events(copied)
        if status.terminal:
            return ProviderOperationSnapshot(state=copied, status=status, events=events)
        stopped = await self._client.run(["stop", "--time", "5", copied.operation_id])
        if stopped.exit_code != 0:
            return ProviderOperationSnapshot(
                state=copied, status=ProviderOperationStatus.UNAVAILABLE
            )
        await self._settle_terminal(
            copied.operation_id,
            self.state_root / copied.operation_id,
            status=ProviderOperationStatus.CANCELLED,
            disposition="cancelled",
        )
        return ProviderOperationSnapshot(state=copied, status=ProviderOperationStatus.CANCELLED)

    async def _recover_connection(
        self, name: str, operation_dir: Path
    ) -> ProviderOperationConnection:
        phase = self._phase(operation_dir)
        authority: dict[str, object] | None = None
        try:
            launch, authority = self._load_bound_operation(name, operation_dir)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            launch = None
        if phase in {"preparing", "prepared", "created", "copied"}:
            try:
                if launch is None or authority is None:
                    raise ValueError("External container operation authority is unavailable.")
                if phase == "preparing":
                    self._prepare_payload(operation_dir / "payload")
                    _write_json(
                        operation_dir / "phase.json",
                        {"schema_version": 1, "phase": "prepared"},
                    )
                await self._advance_admission(
                    name,
                    operation_dir,
                    self._entrypoint_argv(),
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                _write_json(
                    operation_dir / "phase.json",
                    {
                        "schema_version": 1,
                        "phase": "unavailable",
                        "disposition": "unavailable",
                        "diagnostic": (
                            f"External container recovery failed: {type(exc).__name__}."
                        ),
                    },
                )
        return await self._connection(self._state(name, authority=authority))

    def _prepare_payload(self, payload: Path) -> None:
        request_path = payload / "request.json"
        if (
            payload.is_symlink()
            or not payload.is_dir()
            or request_path.is_symlink()
            or not request_path.is_file()
            or request_path.stat().st_size > EXTERNAL_CONTAINER_MAX_INPUT_BYTES
        ):
            raise ValueError("External container preparation state changed identity.")
        payload.chmod(0o700)
        request_path.chmod(0o444)
        _copy_body_snapshot(
            self.body_root,
            payload / "body",
            self.identity.body.content_revision,
        )
        if (
            external_body_file_revision(payload / "body", self.identity.body.private_runtime_path)
            != self.identity.body.private_runtime_revision
        ):
            raise ValueError("Reconstructed private runtime changed identity.")
        payload.chmod(0o555)

    async def _advance_admission(
        self,
        name: str,
        operation_dir: Path,
        argv: list[str],
    ) -> None:
        launch, authority = self._load_bound_operation(name, operation_dir)
        envelope = launch.envelope
        effect_key_digest = authority["idempotency_sha256"]
        launch_request_sha256 = authority["launch_request_sha256"]
        assert type(effect_key_digest) is str
        assert type(launch_request_sha256) is str
        phase = self._phase(operation_dir)
        if phase == "prepared":
            labels_match = await self._labels_match(name, operation_dir)
            if labels_match is False:
                self._mark_admission_unavailable(operation_dir, disposition="identity_mismatch")
                return
            if labels_match is not True:
                create = await self._client.run(
                    self._create_args(
                        name,
                        envelope,
                        argv,
                        idempotency_sha256=effect_key_digest,
                        launch_request_sha256=launch_request_sha256,
                    )
                )
                if create.exit_code != 0:
                    self._mark_admission_unavailable(operation_dir)
                    return
            _write_json(
                operation_dir / "phase.json",
                {"schema_version": 1, "phase": "created"},
            )
            phase = "created"
        if phase == "created":
            labels_match = await self._labels_match(name, operation_dir)
            if labels_match is not True:
                self._mark_admission_unavailable(
                    operation_dir,
                    disposition=("identity_mismatch" if labels_match is False else "unavailable"),
                )
                return
            payload = operation_dir / "payload"
            copied = await self._client.run(["cp", f"{payload}/.", f"{name}:/cayu"])
            if copied.exit_code != 0:
                self._mark_admission_unavailable(operation_dir)
                return
            _write_json(
                operation_dir / "phase.json",
                {"schema_version": 1, "phase": "copied"},
            )
            phase = "copied"
        if phase != "copied":
            return
        labels_match = await self._labels_match(name, operation_dir)
        if labels_match is not True:
            self._mark_admission_unavailable(
                operation_dir,
                disposition=("identity_mismatch" if labels_match is False else "unavailable"),
            )
            return
        inspected = await self._client.run(["inspect", "--format", "{{json .State}}", name])
        try:
            container_state = json.loads(inspected.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            container_state = None
        if inspected.exit_code != 0 or type(container_state) is not dict:
            self._mark_admission_unavailable(operation_dir)
            return
        if container_state.get("Running") is not True and container_state.get("Status") != "exited":
            if container_state.get("Status") != "created":
                self._mark_admission_unavailable(operation_dir)
                return
            started = await self._client.run(["start", name])
            if started.exit_code != 0:
                self._mark_admission_unavailable(operation_dir)
                return
        _write_json(
            operation_dir / "phase.json",
            {"schema_version": 1, "phase": "started"},
        )

    @staticmethod
    def _mark_admission_unavailable(
        operation_dir: Path,
        *,
        disposition: str = "unavailable",
    ) -> None:
        _write_json(
            operation_dir / "phase.json",
            {
                "schema_version": 1,
                "phase": "unavailable",
                "disposition": disposition,
            },
        )

    def _entrypoint_argv(self) -> list[str]:
        return [
            item.replace("{body}", "/cayu/body").replace("{request}", "/cayu/request.json")
            for item in self.identity.body.entrypoint
        ]

    async def _connection(self, state: ProviderOperationState) -> ProviderOperationConnection:
        status, _ = await self._status_and_events(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            while True:
                current, retained = await self._status_and_events(state)
                if current.terminal:
                    for event in retained:
                        yield event
                    return
                await asyncio.sleep(self.poll_seconds)

        return ProviderOperationConnection(state=state, status=status, events=events())

    async def _status_and_events(
        self,
        state: ProviderOperationState,
    ) -> tuple[ProviderOperationStatus, tuple[ModelStreamEvent, ...]]:
        operation_dir = self.state_root / state.operation_id
        try:
            _, authority = self._load_bound_operation(state.operation_id, operation_dir)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "identity_mismatch", ProviderOperationStatus.UNAVAILABLE
            )
        if not self._state_matches_authority(state, authority):
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "identity_mismatch", ProviderOperationStatus.UNAVAILABLE
            )
        terminal = self._retained_terminal(state, operation_dir)
        if terminal is not None:
            await self._cleanup_terminal_container(state.operation_id, operation_dir)
            return terminal
        phase = self._phase(operation_dir)
        if phase == "cancelled":
            return ProviderOperationStatus.CANCELLED, self._error_events(
                state, "cancelled", ProviderOperationStatus.CANCELLED
            )
        if phase == "created":
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "incomplete", ProviderOperationStatus.UNAVAILABLE
            )
        if phase == "copied":
            return ProviderOperationStatus.IN_PROGRESS, ()
        if phase == "unavailable":
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state,
                self._phase_disposition(operation_dir) or "unavailable",
                ProviderOperationStatus.UNAVAILABLE,
            )
        if phase is None:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        labels_match = await self._labels_match(state.operation_id, operation_dir)
        if labels_match is False:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "identity_mismatch", ProviderOperationStatus.UNAVAILABLE
            )
        if labels_match is None:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        inspected = await self._client.run(
            ["inspect", "--format", "{{json .State}}", state.operation_id]
        )
        if inspected.exit_code != 0 or len(inspected.stdout) > 64 << 10:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        try:
            container_state = json.loads(inspected.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            container_state = None
        if type(container_state) is not dict:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        if container_state.get("Running") is True:
            return ProviderOperationStatus.IN_PROGRESS, ()
        if container_state.get("Status") in {"created", "restarting"}:
            return ProviderOperationStatus.IN_PROGRESS, ()
        if container_state.get("Status") != "exited":
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unavailable", ProviderOperationStatus.UNAVAILABLE
            )
        exit_code = container_state.get("ExitCode")
        if exit_code != 0:
            disposition = "oom_killed" if container_state.get("OOMKilled") is True else "failed"
            await self._settle_terminal(
                state.operation_id,
                operation_dir,
                status=ProviderOperationStatus.FAILED,
                disposition=disposition,
            )
            return ProviderOperationStatus.FAILED, self._error_events(
                state, disposition, ProviderOperationStatus.FAILED
            )
        logs = await self._client.run(["logs", state.operation_id])
        if logs.exit_code != 0 or len(logs.stdout) > EXTERNAL_CONTAINER_MAX_OUTPUT_BYTES:
            await self._settle_terminal(
                state.operation_id,
                operation_dir,
                status=ProviderOperationStatus.FAILED,
                disposition="incomplete",
            )
            return ProviderOperationStatus.FAILED, self._error_events(
                state, "incomplete", ProviderOperationStatus.FAILED
            )
        try:
            output = ExternalContainerOutputV1.model_validate_json(logs.stdout)
        except ValueError:
            await self._settle_terminal(
                state.operation_id,
                operation_dir,
                status=ProviderOperationStatus.FAILED,
                disposition="incomplete",
            )
            return ProviderOperationStatus.FAILED, self._error_events(
                state, "incomplete", ProviderOperationStatus.FAILED
            )
        await self._settle_terminal(
            state.operation_id,
            operation_dir,
            status=ProviderOperationStatus.COMPLETED,
            disposition="completed",
            output=output,
        )
        return ProviderOperationStatus.COMPLETED, self._completed_events(state, output)

    def _retained_terminal(
        self,
        state: ProviderOperationState,
        operation_dir: Path,
    ) -> tuple[ProviderOperationStatus, tuple[ModelStreamEvent, ...]] | None:
        path = operation_dir / "terminal.json"
        if not path.exists():
            return None
        try:
            _, authority = self._load_bound_operation(state.operation_id, operation_dir)
            if not self._state_matches_authority(state, authority):
                return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                    state, "identity_mismatch", ProviderOperationStatus.UNAVAILABLE
                )
            document = _read_json(path, maximum_bytes=2 << 20)
            status = ProviderOperationStatus(document["status"])
            disposition = document["disposition"]
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        terminal_identity = {
            key: document.get(key)
            for key in (
                "target_revision",
                "trial_revision",
                "environment_revision",
                "idempotency_sha256",
                "launch_request_sha256",
            )
        }
        if terminal_identity != self._receipt_identity(authority):
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "identity_mismatch", ProviderOperationStatus.UNAVAILABLE
            )
        if type(disposition) is not str or not status.terminal:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        if status is ProviderOperationStatus.COMPLETED:
            try:
                output = ExternalContainerOutputV1.model_validate(document["output"])
            except (KeyError, ValueError):
                return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                    state, "unknown", ProviderOperationStatus.UNAVAILABLE
                )
            return status, self._completed_events(state, output)
        if status not in {ProviderOperationStatus.FAILED, ProviderOperationStatus.CANCELLED}:
            return ProviderOperationStatus.UNAVAILABLE, self._error_events(
                state, "unknown", ProviderOperationStatus.UNAVAILABLE
            )
        return status, self._error_events(state, disposition, status)

    async def _settle_terminal(
        self,
        name: str,
        operation_dir: Path,
        *,
        status: ProviderOperationStatus,
        disposition: str,
        output: ExternalContainerOutputV1 | None = None,
    ) -> None:
        _, authority = self._load_bound_operation(name, operation_dir)
        document: dict[str, object] = {
            "schema_version": 2,
            **self._receipt_identity(authority),
            "status": status.value,
            "disposition": disposition,
        }
        if output is not None:
            document["output"] = output.model_dump(mode="json")
        _write_json_once(operation_dir / "terminal.json", document)
        _write_json(
            operation_dir / "phase.json",
            {"schema_version": 1, "phase": "terminal"},
        )
        await self._cleanup_terminal_container(name, operation_dir)

    async def _cleanup_terminal_container(self, name: str, operation_dir: Path) -> None:
        cleanup_path = operation_dir / "cleanup.json"
        try:
            cleanup = _read_json(cleanup_path, maximum_bytes=16 << 10)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            cleanup = None
        if cleanup == {"schema_version": 1, "state": "removed"}:
            return
        removed = await self._client.run(["rm", "-f", name])
        if removed.exit_code == 0:
            _write_json(
                cleanup_path,
                {"schema_version": 1, "state": "removed"},
            )

    def _create_args(
        self,
        name: str,
        envelope: ExternalTrialEnvelopeV1,
        argv: list[str],
        *,
        idempotency_sha256: str,
        launch_request_sha256: str,
    ) -> list[str]:
        return [
            "create",
            "--name",
            name,
            "--runtime",
            self.runtime,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--user",
            "65532:65532",
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs",
            "/workspace:rw,nosuid,nodev,size=256m",
            "--log-opt",
            "max-size=1m",
            "--log-opt",
            "max-file=1",
            "--label",
            f"{_CONTAINER_LABEL_PREFIX}.target={self.identity.revision}",
            "--label",
            f"{_CONTAINER_LABEL_PREFIX}.trial={envelope.trial.revision}",
            "--label",
            f"{_CONTAINER_LABEL_PREFIX}.environment={self.identity.environment_revision}",
            "--label",
            f"{_CONTAINER_LABEL_PREFIX}.idempotency={idempotency_sha256}",
            "--label",
            f"{_CONTAINER_LABEL_PREFIX}.request={launch_request_sha256}",
            "--entrypoint",
            argv[0],
            "--",
            self.image,
            *argv[1:],
        ]

    @staticmethod
    def _completed_events(
        state: ProviderOperationState,
        output: ExternalContainerOutputV1,
    ) -> tuple[ModelStreamEvent, ...]:
        events = (
            ModelStreamEvent.text_delta(output.output, recovery_metadata={"cursor": 1}),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "external_candidate_diagnostics": {
                        "usage_trust": "candidate_reported_untrusted",
                        "reported_usage": output.usage.model_dump(mode="json"),
                    },
                },
                recovery_metadata={"cursor": 2},
            ),
        )
        return events[state.recovery_metadata.cursor or 0 :]

    @staticmethod
    def _error_events(
        state: ProviderOperationState,
        disposition: str,
        operation_status: ProviderOperationStatus,
    ) -> tuple[ModelStreamEvent, ...]:
        if (state.recovery_metadata.cursor or 0) >= 1:
            return ()
        message = f"External container ended with disposition {disposition}."
        error = ModelProviderError(
            message,
            provider="cayu-external-process",
            error_type="ExternalContainerDisposition",
            error_code=f"external_container_{disposition}",
            retryable=False,
        )
        return (
            ModelStreamEvent.error(
                message,
                cause=error,
                provider_operation_status=operation_status,
                recovery_metadata={"cursor": 1},
            ),
        )

    async def _labels_match(self, name: str, operation_dir: Path) -> bool | None:
        try:
            _, authority = self._load_bound_operation(name, operation_dir)
            expected = {
                f"{_CONTAINER_LABEL_PREFIX}.target": authority["target_revision"],
                f"{_CONTAINER_LABEL_PREFIX}.trial": authority["trial_revision"],
                f"{_CONTAINER_LABEL_PREFIX}.environment": authority["environment_revision"],
                f"{_CONTAINER_LABEL_PREFIX}.idempotency": authority["idempotency_sha256"],
                f"{_CONTAINER_LABEL_PREFIX}.request": authority["launch_request_sha256"],
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if any(type(value) is not str for value in expected.values()):
            return None
        inspected = await self._client.run(["inspect", "--format", "{{json .Config.Labels}}", name])
        if inspected.exit_code != 0 or len(inspected.stdout) > 64 << 10:
            return None
        try:
            labels = json.loads(inspected.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if type(labels) is not dict:
            return None
        return all(labels.get(key) == value for key, value in expected.items())

    @staticmethod
    def _state(
        name: str,
        *,
        authority: dict[str, object] | None,
    ) -> ProviderOperationState:
        opaque = (
            {}
            if authority is None
            else ExternalContainerOperationAdapter._receipt_identity(authority)
        )
        return ProviderOperationState(
            operation_id=name,
            stream_protocol=EXTERNAL_CONTAINER_STREAM_PROTOCOL,
            recovery_metadata=ProviderOperationRecoveryMetadata(cursor=0, opaque=opaque),
        )

    @staticmethod
    def _validated_state(state: ProviderOperationState) -> ProviderOperationState:
        if type(state) is not ProviderOperationState:
            raise TypeError("state must be an exact ProviderOperationState.")
        if (
            state.stream_protocol != EXTERNAL_CONTAINER_STREAM_PROTOCOL
            or not state.operation_id.startswith(_CONTAINER_NAME_PREFIX)
        ):
            raise ValueError("Provider operation state does not belong to this adapter.")
        return ProviderOperationState.model_validate(state.model_dump(mode="python"))

    @staticmethod
    def _phase(operation_dir: Path) -> str | None:
        try:
            phase = _read_json(operation_dir / "phase.json", maximum_bytes=16 << 10).get("phase")
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return phase if type(phase) is str else None

    @staticmethod
    def _phase_disposition(operation_dir: Path) -> str | None:
        try:
            disposition = _read_json(operation_dir / "phase.json", maximum_bytes=16 << 10).get(
                "disposition"
            )
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return disposition if type(disposition) is str else None

    def _bind_start_alias(
        self,
        start_key_digest: str,
        *,
        name: str,
        authority: dict[str, object],
    ) -> None:
        document = {
            "schema_version": 2,
            "operation_id": name,
            **self._receipt_identity(authority),
        }
        path = self._start_alias_root / f"{start_key_digest}.json"
        temporary = self._start_alias_root / (f".{start_key_digest}.{secrets.token_hex(8)}.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                retained_name, retained_authority = self._start_alias(start_key_digest)
                if retained_name != name or retained_authority != authority:
                    raise ValueError("External container start alias changed identity.") from None
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def _start_alias(self, start_key_digest: str) -> tuple[str, dict[str, object]]:
        try:
            value = _read_json(
                self._start_alias_root / f"{start_key_digest}.json",
                maximum_bytes=16 << 10,
            )
            name = value["operation_id"]
            trial_revision = value["trial_revision"]
            effect_key_digest = value["idempotency_sha256"]
            launch_request_sha256 = value["launch_request_sha256"]
        except (KeyError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("External container start identity is unavailable.") from None
        if (
            type(name) is not str
            or type(trial_revision) is not str
            or type(effect_key_digest) is not str
            or type(launch_request_sha256) is not str
            or name != _container_name(trial_revision)
            or effect_key_digest != sha256(trial_revision.encode("utf-8")).hexdigest()
        ):
            raise ValueError("External container start identity is invalid.")
        authority = self._authority_document(
            trial_revision,
            effect_key_digest=effect_key_digest,
            launch_request_sha256=launch_request_sha256,
        )
        expected = {
            "schema_version": 2,
            "operation_id": name,
            **self._receipt_identity(authority),
        }
        if value != expected:
            raise ValueError("External container start identity is invalid.")
        return name, authority

    def _load_bound_operation(
        self,
        name: str,
        operation_dir: Path,
    ) -> tuple[ExternalContainerLaunchRequestV1, dict[str, object]]:
        request_path = operation_dir / "payload" / "request.json"
        if (
            operation_dir.name != name
            or operation_dir.is_symlink()
            or not operation_dir.is_dir()
            or request_path.is_symlink()
            or not request_path.is_file()
            or request_path.stat().st_size > EXTERNAL_CONTAINER_MAX_INPUT_BYTES
        ):
            raise ValueError("External container operation authority is unavailable.")
        try:
            request_bytes = request_path.read_bytes()
            launch = ExternalContainerLaunchRequestV1.model_validate_json(request_bytes)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("External container operation authority is unavailable.") from None
        canonical_request = canonical_durable_json_bytes(
            launch.model_dump(mode="json"),
            "external container launch request",
        )
        if request_bytes != canonical_request:
            raise ValueError("External container launch request changed identity.")
        trial_revision = launch.envelope.trial.revision
        if (
            launch.envelope.trial.target_revision != self.identity.revision
            or name != _container_name(trial_revision)
        ):
            raise ValueError("External container launch request changed identity.")
        effect_key_digest = sha256(trial_revision.encode("utf-8")).hexdigest()
        expected = self._authority_document(
            trial_revision,
            effect_key_digest=effect_key_digest,
            launch_request_sha256=sha256(request_bytes).hexdigest(),
        )
        try:
            authority = _read_json(operation_dir / "authority.json", maximum_bytes=16 << 10)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("External container operation authority is unavailable.") from None
        if authority != expected:
            raise ValueError("External container operation authority changed identity.")
        self._require_key(operation_dir, effect_key_digest)
        return launch, expected

    def _authority_document(
        self,
        trial_revision: str,
        *,
        effect_key_digest: str,
        launch_request_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "target_revision": self.identity.revision,
            "trial_revision": trial_revision,
            "environment_revision": self.identity.environment_revision,
            "idempotency_sha256": effect_key_digest,
            "launch_request_sha256": launch_request_sha256,
        }

    @staticmethod
    def _receipt_identity(authority: dict[str, object]) -> dict[str, object]:
        return {
            key: authority[key]
            for key in (
                "target_revision",
                "trial_revision",
                "environment_revision",
                "idempotency_sha256",
                "launch_request_sha256",
            )
        }

    @staticmethod
    def _state_matches_authority(
        state: ProviderOperationState,
        authority: dict[str, object],
    ) -> bool:
        return (
            state.recovery_metadata.opaque
            == ExternalContainerOperationAdapter._receipt_identity(authority)
        )

    @staticmethod
    def _require_key(operation_dir: Path, expected: str) -> None:
        path = operation_dir / "idempotency.sha256"
        try:
            if path.is_symlink():
                raise OSError
            observed = path.read_text(encoding="ascii")
        except OSError:
            raise ValueError("External container operation identity is unavailable.") from None
        if observed != expected:
            raise ValueError("External container operation idempotency identity changed.")


__all__ = [
    "EXTERNAL_CONTAINER_MAX_INPUT_BYTES",
    "EXTERNAL_CONTAINER_MAX_OUTPUT_BYTES",
    "EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION",
    "EXTERNAL_CONTAINER_RUNNER_REVISION",
    "EXTERNAL_CONTAINER_STREAM_PROTOCOL",
    "ExternalContainerLaunchRequestV1",
    "ExternalContainerOperationAdapter",
    "ExternalContainerOutputV1",
    "ExternalContainerUsageV1",
    "external_container_environment_revision",
]
