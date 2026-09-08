"""Explicit protected browser transport deployment configuration."""

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretBytes, field_validator


class BrowserControlServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operator_origin: str
    signing_key: SecretBytes = Field(repr=False, exclude=True)

    @field_validator("operator_origin")
    @classmethod
    def exact_origin(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            valid = (
                value.isascii()
                and value == value.strip()
                and not any(ord(char) < 33 for char in value)
                and parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
                and parsed.port != 0
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Browser control requires an exact HTTPS operator origin.")
        return value

    @field_validator("signing_key")
    @classmethod
    def dedicated_key(cls, value: SecretBytes) -> SecretBytes:
        if len(value.get_secret_value()) != 32:
            raise ValueError("Browser control requires a dedicated 32-byte signing key.")
        return value
