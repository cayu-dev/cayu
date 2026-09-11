"""Private accepted coding configuration; not authentication or dispatch authority."""

import json

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from cayu import CodingSettlementPolicy


def invalid_request() -> ValueError:
    return ValueError("Invalid or changed maintenance request configuration.")


def default_maintenance_settlement() -> CodingSettlementPolicy:
    return CodingSettlementPolicy(
        required_checks=("format", "independent-range-probe", "lint", "test"),
        reviewer_required=False,
        human_approval_required=False,
    )


def bounded_text(value: object, *, bound: int = 512) -> str:
    if type(value) is not str:
        raise invalid_request()
    try:
        if not value.strip() or len(value.encode("utf-8")) > bound or "\x00" in value:
            raise invalid_request()
    except UnicodeError:
        raise invalid_request() from None
    return value


class MaintenanceAcceptedRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: StrictStr
    instruction: StrictStr
    repository_root: StrictStr
    source_workspace_id: StrictStr
    source_origin_id: StrictStr
    source_destination_id: StrictStr
    artifact_store_id: StrictStr
    toolchain_profile_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    corpus_fingerprint: StrictStr
    probe_fingerprint: StrictStr
    base_revision: StrictStr
    settlement_json: StrictStr

    @field_validator("*", mode="before")
    @classmethod
    def validate_text(cls, value, info):
        bound = (
            4096
            if info.field_name in {"instruction", "repository_root", "settlement_json"}
            else 512
        )
        return bounded_text(value, bound=bound)

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value):
        if value != "maintenance.accepted.v1":
            raise invalid_request()
        return value

    @field_validator(
        "toolchain_profile_fingerprint",
        "corpus_fingerprint",
        "probe_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value):
        if (
            len(value) != 71
            or not value.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in value[7:])
        ):
            raise invalid_request()
        return value

    @field_validator("execution_profile_fingerprint")
    @classmethod
    def validate_execution_profile(cls, value):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise invalid_request()
        return value

    @field_validator("base_revision")
    @classmethod
    def validate_base(cls, value):
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise invalid_request()
        return value


def copy_request(value: MaintenanceAcceptedRequest) -> MaintenanceAcceptedRequest:
    if type(value) is not MaintenanceAcceptedRequest:
        raise invalid_request()
    return MaintenanceAcceptedRequest.model_validate(
        {name: getattr(value, name) for name in MaintenanceAcceptedRequest.model_fields}
    )


def encode_request(value: MaintenanceAcceptedRequest) -> str:
    return copy_request(value).model_dump_json()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise invalid_request()
        result[key] = value
    return result


def decode_request(value: str) -> MaintenanceAcceptedRequest:
    value = bounded_text(value, bound=65536)
    try:
        data = json.loads(value, object_pairs_hook=_unique_object)
        return MaintenanceAcceptedRequest.model_validate(data)
    except (ValueError, RecursionError):
        raise invalid_request() from None
