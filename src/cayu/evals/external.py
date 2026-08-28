"""Contracts for evaluating immutable agent bodies outside the evaluator process.

The public models in this module contain identity and authority-free request facts
only.  Container handles, body resolvers, credentials, scorers, and effect stores
remain process-local capabilities owned by a trusted target adapter.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import Message, MessageRole, TextPart, detach_message
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    _preflight_provider_portable_messages,
)
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationMode,
)

EXTERNAL_PROCESS_PROTOCOL_VERSION = "cayu.external-process.v1"
EXTERNAL_TRIAL_ENVELOPE_PREFIX = "cayu-external-trial-v1:"
EXTERNAL_TARGET_MAX_ENTRYPOINT_PARTS = 32
EXTERNAL_TARGET_MAX_IDENTITY_CHARS = 512
EXTERNAL_BODY_MAX_BYTES = 32 << 20
EXTERNAL_BODY_MAX_FILES = 512
_EXTERNAL_BODY_REVISION_DOMAIN = "cayu.external-body-release.v1"
_EXTERNAL_BODY_CONTENT_DOMAIN = "cayu.external-body-content.v1"
_EXTERNAL_TARGET_REVISION_DOMAIN = "cayu.external-process-target.v1"
_EXTERNAL_TRIAL_REVISION_DOMAIN = "cayu.external-trial-identity.v1"


def _content_revision(value: object, field_name: str) -> str:
    digest = sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()
    return f"sha256:{digest}"


def _sha256_revision(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 revision.")
    return value


def _bounded_identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > EXTERNAL_TARGET_MAX_IDENTITY_CHARS:
        raise ValueError(
            f"{field_name} cannot exceed {EXTERNAL_TARGET_MAX_IDENTITY_CHARS} characters."
        )
    return value


def _body_relative_path(value: str, field_name: str) -> str:
    value = _bounded_identity(value, field_name)
    relative = PurePosixPath(value)
    if (
        not relative.parts
        or relative.is_absolute()
        or "\\" in value
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ValueError(f"{field_name} must be a canonical relative POSIX path.")
    return value


def _external_body_manifest(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    if not root_path.is_absolute() or not root_path.is_dir() or root_path.is_symlink():
        raise ValueError("External body root must be an absolute, non-symlink directory.")
    files: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("External body snapshots cannot contain symbolic links.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("External body snapshots can contain only regular files.")
        relative = path.relative_to(root_path).as_posix()
        size = path.stat().st_size
        total_bytes += size
        if len(files) >= EXTERNAL_BODY_MAX_FILES or total_bytes > EXTERNAL_BODY_MAX_BYTES:
            raise ValueError("External body snapshot exceeds its file or byte limit.")
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        files.append(
            {
                "path": relative,
                "size": size,
                # Write/read permissions are deployment details. Executability
                # changes launch semantics and is therefore content identity.
                "executable_mode": os.stat(path, follow_symlinks=False).st_mode & 0o111,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ValueError("External body snapshot must contain at least one file.")
    return {"schema_version": 1, "files": files}


def external_body_content_revision(root: str | Path) -> str:
    """Return the canonical content revision of one bounded multi-file body."""

    return _content_revision(_external_body_manifest(root), _EXTERNAL_BODY_CONTENT_DOMAIN)


def external_body_file_revision(root: str | Path, relative_path: str) -> str:
    """Return a content revision for one required file inside a body snapshot."""

    root_path = Path(root)
    relative = _body_relative_path(relative_path, "relative_path")
    path = root_path.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("External body file must be a regular non-symlink file.")
    digest = sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


class _ExternalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class ExternalBodyReleaseV1(_ExternalModel):
    """Content identity and launch contract for one immutable packaged body.

    ``content_revision`` commits the complete multi-file body.  The private
    runtime and launch protocol are named independently so replacing the bundled
    runtime cannot hide behind an otherwise unchanged target identity.
    """

    schema_version: Literal[1] = 1
    revision: StrictStr
    content_revision: StrictStr
    private_runtime_path: StrictStr
    private_runtime_revision: StrictStr
    launch_protocol_revision: StrictStr
    entrypoint: tuple[StrictStr, ...] = Field(
        min_length=1,
        max_length=EXTERNAL_TARGET_MAX_ENTRYPOINT_PARTS,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "revision",
        "content_revision",
        "private_runtime_revision",
        "launch_protocol_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("private_runtime_path")
    @classmethod
    def validate_private_runtime_path(cls, value: str, info) -> str:
        return _body_relative_path(value, info.field_name)

    @field_validator("entrypoint", mode="before")
    @classmethod
    def validate_entrypoint_sequence(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("entrypoint must be an ordered array.")
        return value

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _bounded_identity(item, f"entrypoint[{index}]") for index, item in enumerate(value)
        )

    @model_validator(mode="after")
    def validate_revision(self) -> ExternalBodyReleaseV1:
        expected = _content_revision(
            self.model_dump(mode="json", exclude={"revision"}),
            _EXTERNAL_BODY_REVISION_DOMAIN,
        )
        if self.revision != expected:
            raise ValueError("External body release revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        content_revision: str,
        private_runtime_path: str,
        private_runtime_revision: str,
        launch_protocol_revision: str,
        entrypoint: Sequence[str],
    ) -> ExternalBodyReleaseV1:
        if isinstance(entrypoint, str | bytes) or not isinstance(entrypoint, Sequence):
            raise TypeError("entrypoint must be an ordered sequence of strings.")
        material = {
            "schema_version": 1,
            "content_revision": content_revision,
            "private_runtime_path": private_runtime_path,
            "private_runtime_revision": private_runtime_revision,
            "launch_protocol_revision": launch_protocol_revision,
            "entrypoint": list(entrypoint),
        }
        return cls(
            revision=_content_revision(material, _EXTERNAL_BODY_REVISION_DOMAIN),
            content_revision=content_revision,
            private_runtime_path=private_runtime_path,
            private_runtime_revision=private_runtime_revision,
            launch_protocol_revision=launch_protocol_revision,
            entrypoint=tuple(entrypoint),
        )

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        private_runtime_path: str,
        launch_protocol_revision: str,
        entrypoint: Sequence[str],
    ) -> ExternalBodyReleaseV1:
        """Pin a complete body plus its separately identified bundled runtime."""

        return cls.create(
            content_revision=external_body_content_revision(root),
            private_runtime_path=private_runtime_path,
            private_runtime_revision=external_body_file_revision(root, private_runtime_path),
            launch_protocol_revision=launch_protocol_revision,
            entrypoint=entrypoint,
        )


class ExternalProcessTargetIdentityV1(_ExternalModel):
    """Independently pinned identities for one trusted external target."""

    schema_version: Literal[1] = 1
    revision: StrictStr
    body: ExternalBodyReleaseV1
    evaluator_runtime_revision: StrictStr
    target_implementation_revision: StrictStr
    runner_revision: StrictStr
    environment_revision: StrictStr
    reset_contract_revision: StrictStr
    evidence_policy_revision: StrictStr

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "revision",
        "evaluator_runtime_revision",
        "target_implementation_revision",
        "runner_revision",
        "environment_revision",
        "reset_contract_revision",
        "evidence_policy_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("body", mode="before")
    @classmethod
    def copy_body(cls, value: object) -> object:
        if type(value) is ExternalBodyReleaseV1:
            return value.model_dump(mode="json")
        return value

    @model_validator(mode="after")
    def validate_revision(self) -> ExternalProcessTargetIdentityV1:
        expected = _content_revision(
            self.model_dump(mode="json", exclude={"revision"}),
            _EXTERNAL_TARGET_REVISION_DOMAIN,
        )
        if self.revision != expected:
            raise ValueError("External process target revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        body: ExternalBodyReleaseV1,
        evaluator_runtime_revision: str,
        target_implementation_revision: str,
        runner_revision: str,
        environment_revision: str,
        reset_contract_revision: str,
        evidence_policy_revision: str,
    ) -> ExternalProcessTargetIdentityV1:
        if type(body) is not ExternalBodyReleaseV1:
            raise TypeError("body must be an exact ExternalBodyReleaseV1.")
        material = {
            "schema_version": 1,
            "body": body.model_dump(mode="json"),
            "evaluator_runtime_revision": evaluator_runtime_revision,
            "target_implementation_revision": target_implementation_revision,
            "runner_revision": runner_revision,
            "environment_revision": environment_revision,
            "reset_contract_revision": reset_contract_revision,
            "evidence_policy_revision": evidence_policy_revision,
        }
        return cls(
            revision=_content_revision(material, _EXTERNAL_TARGET_REVISION_DOMAIN),
            **material,
        )


class ExternalTrialIdentityV1(_ExternalModel):
    """Exact native run/corpus/suite/case/trial identity fixed before dispatch."""

    schema_version: Literal[1] = 1
    revision: StrictStr
    native_run_id: StrictStr
    target_key: StrictStr
    target_revision: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    case_id: StrictStr
    case_revision: StrictStr
    trial_number: StrictInt = Field(ge=1, le=100)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "revision",
        "target_revision",
        "corpus_revision",
        "suite_revision",
        "case_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("native_run_id", "target_key", "suite_id", "case_id")
    @classmethod
    def validate_identities(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @model_validator(mode="after")
    def validate_revision(self) -> ExternalTrialIdentityV1:
        expected = _content_revision(
            self.model_dump(mode="json", exclude={"revision"}),
            _EXTERNAL_TRIAL_REVISION_DOMAIN,
        )
        if self.revision != expected:
            raise ValueError("External trial revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        native_run_id: str,
        target_key: str,
        target_revision: str,
        corpus_revision: str,
        suite_id: str,
        suite_revision: str,
        case_id: str,
        case_revision: str,
        trial_number: int,
    ) -> ExternalTrialIdentityV1:
        material = {
            "schema_version": 1,
            "native_run_id": native_run_id,
            "target_key": target_key,
            "target_revision": target_revision,
            "corpus_revision": corpus_revision,
            "suite_id": suite_id,
            "suite_revision": suite_revision,
            "case_id": case_id,
            "case_revision": case_revision,
            "trial_number": trial_number,
        }
        return cls(
            revision=_content_revision(material, _EXTERNAL_TRIAL_REVISION_DOMAIN),
            **material,
        )


class OpaqueExternalCaseRefV1(_ExternalModel):
    """Authority-free alias for case material retained by a trusted adapter."""

    schema_version: Literal[1] = 1
    id: StrictStr
    revision: StrictStr

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


class ExternalTrialEnvelopeV1(_ExternalModel):
    """Runtime-authored marker carried to the trusted external provider only."""

    schema_version: Literal[1] = 1
    protocol: Literal["cayu.external-process.v1"] = EXTERNAL_PROCESS_PROTOCOL_VERSION
    trial: ExternalTrialIdentityV1
    opaque_case_ref: OpaqueExternalCaseRefV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("trial", mode="before")
    @classmethod
    def copy_trial(cls, value: object) -> object:
        if type(value) is ExternalTrialIdentityV1:
            return value.model_dump(mode="json")
        return value

    @field_validator("opaque_case_ref", mode="before")
    @classmethod
    def copy_opaque_case_ref(cls, value: object) -> object:
        if type(value) is OpaqueExternalCaseRefV1:
            return value.model_dump(mode="json")
        return value

    def message(self) -> Message:
        document = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return Message.text(
            MessageRole.SYSTEM,
            EXTERNAL_TRIAL_ENVELOPE_PREFIX + document,
        )


def with_external_trial_envelope(
    messages: Sequence[Message],
    envelope: ExternalTrialEnvelopeV1,
) -> list[Message]:
    """Return detached messages prefixed by one exact runtime-owned envelope."""

    if isinstance(messages, str | bytes) or not isinstance(messages, Sequence):
        raise TypeError("messages must be an ordered sequence of Messages.")
    if type(envelope) is not ExternalTrialEnvelopeV1:
        raise TypeError("envelope must be an exact ExternalTrialEnvelopeV1.")
    copied = [detach_message(message) for message in messages]
    if any(_external_envelope_text(message) is not None for message in copied):
        raise ValueError("External process messages already contain a trial envelope.")
    return [envelope.message(), *copied]


def external_trial_envelope_from_request(
    request: ModelRequest,
    *,
    expected_target_revision: str,
) -> tuple[ExternalTrialEnvelopeV1, ModelRequest]:
    """Extract and remove the one trusted marker from a provider request."""

    if type(request) is not ModelRequest:
        raise TypeError("request must be an exact ModelRequest.")
    expected_target_revision = _sha256_revision(
        expected_target_revision,
        "expected_target_revision",
    )
    matches: list[tuple[int, str]] = []
    for index, message in enumerate(request.messages):
        text = _external_envelope_text(message)
        if text is not None:
            matches.append((index, text))
    if len(matches) != 1:
        raise ValueError("External process requests require exactly one trial envelope.")
    index, document = matches[0]
    try:
        envelope = ExternalTrialEnvelopeV1.model_validate_json(document)
    except ValueError:
        raise ValueError("External process request trial envelope is invalid.") from None
    if envelope.trial.target_revision != expected_target_revision:
        raise ValueError("External process request target identity changed after admission.")
    remaining = [
        detach_message(message)
        for message_index, message in enumerate(request.messages)
        if message_index != index
    ]
    return envelope, request.model_copy(update={"messages": remaining}, deep=True)


def _external_envelope_text(message: Message) -> str | None:
    if type(message) is not Message or message.role is not MessageRole.SYSTEM:
        return None
    if len(message.content) != 1 or type(message.content[0]) is not TextPart:
        return None
    text = require_durable_text(message.content[0].text, "external trial envelope")
    if not text.startswith(EXTERNAL_TRIAL_ENVELOPE_PREFIX):
        return None
    return text[len(EXTERNAL_TRIAL_ENVELOPE_PREFIX) :]


class ExternalProcessModelProvider(ModelProvider):
    """One-step provider façade over a reconnectable trusted target adapter.

    The provider owns no scheduling or retry state.  Cayu's ordinary provider
    operation machinery persists the adapter's opaque operation identity and
    reconnects it after process loss.
    """

    name = "cayu-external-process"

    def __init__(
        self,
        *,
        identity: ExternalProcessTargetIdentityV1,
        operations: ProviderOperationAdapter,
    ) -> None:
        if type(identity) is not ExternalProcessTargetIdentityV1:
            raise TypeError("identity must be an exact ExternalProcessTargetIdentityV1.")
        if not isinstance(operations, ProviderOperationAdapter):
            raise TypeError("operations must implement ProviderOperationAdapter.")
        self.identity = ExternalProcessTargetIdentityV1.model_validate(
            identity.model_dump(mode="json")
        )
        self._operations = operations

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="cayu.external-process-provider",
            behavior_version=EXTERNAL_PROCESS_PROTOCOL_VERSION,
            implementation_version=self.identity.revision,
        )

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self._operations

    def preflight_model_target(self, *, model: str) -> None:
        if model != EXTERNAL_PROCESS_PROTOCOL_VERSION:
            raise ValueError("External process target model does not match its launch protocol.")

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        self.preflight_model_target(model=model)
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=False,
            supports_tool_definitions=False,
            supports_file_attachments=True,
        )

    async def stream(self, request: ModelRequest):
        del request
        raise RuntimeError(
            "External process providers execute only through reconnectable operations."
        )
        yield ModelStreamEvent.text_delta("")  # pragma: no cover


__all__ = [
    "EXTERNAL_BODY_MAX_BYTES",
    "EXTERNAL_BODY_MAX_FILES",
    "EXTERNAL_PROCESS_PROTOCOL_VERSION",
    "EXTERNAL_TRIAL_ENVELOPE_PREFIX",
    "ExternalBodyReleaseV1",
    "ExternalProcessModelProvider",
    "ExternalProcessTargetIdentityV1",
    "ExternalTrialEnvelopeV1",
    "ExternalTrialIdentityV1",
    "OpaqueExternalCaseRefV1",
    "external_body_content_revision",
    "external_body_file_revision",
    "external_trial_envelope_from_request",
    "with_external_trial_envelope",
]
