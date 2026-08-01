from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
from collections.abc import Mapping
from typing import Any, cast

import pytest
from pydantic import SecretStr

from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_MAX_KEYS,
    PUBLIC_AUTHORITY_ALIAS_PREFIX,
    PUBLIC_AUTHORITY_ALIAS_VERSION,
    ParsedPublicAuthorityAlias,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    parse_public_authority_alias,
    public_authority_alias_codec_from_environment,
    public_authority_alias_is_reserved,
)


def _key(byte: int) -> SecretStr:
    return SecretStr(base64.urlsafe_b64encode(bytes([byte]) * 32).decode().rstrip("="))


def test_codec_produces_canonical_field_and_session_scoped_hmac_aliases() -> None:
    keyring = PublicAuthorityAliasKeyring(active_key_id="primary", keys={"primary": _key(1)})
    codec = PublicAuthorityAliasCodec(keyring)

    session_alias = codec.encode("private-session", field_name="session_id")
    interaction_alias = codec.encode(
        "private-interaction",
        field_name="interaction_id",
        session_id="private-session",
    )

    assert session_alias.startswith(
        f"{PUBLIC_AUTHORITY_ALIAS_PREFIX}{PUBLIC_AUTHORITY_ALIAS_VERSION}.primary.session_id."
    )
    parsed = codec.parse(interaction_alias)
    assert parsed == ParsedPublicAuthorityAlias(
        version=PUBLIC_AUTHORITY_ALIAS_VERSION,
        key_id="primary",
        field_name="interaction_id",
        tag=interaction_alias.rsplit(".", 1)[1],
    )
    assert parsed is not None
    assert len(parsed.tag) == 43
    assert "=" not in parsed.tag
    assert codec.matches(session_alias, "private-session", field_name="session_id")
    assert codec.matches(
        interaction_alias,
        "private-interaction",
        field_name="interaction_id",
        session_id="private-session",
    )

    assert not codec.matches(session_alias, "other-session", field_name="session_id")
    assert not codec.matches(session_alias, "private-session", field_name="interaction_id")
    assert not codec.matches(
        interaction_alias,
        "private-interaction",
        field_name="interaction_id",
        session_id="other-session",
    )
    assert (
        codec.encode(
            "private-interaction",
            field_name="interaction_id",
            session_id="other-session",
        )
        != interaction_alias
    )


def test_codec_hmac_input_is_canonical_and_not_an_unkeyed_digest() -> None:
    encoded_key = _key(7)
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="key-1", keys={"key-1": encoded_key})
    )

    alias = codec.encode("private-value", field_name="session_id")
    tag = alias.rsplit(".", 1)[1]
    decoded_tag = base64.urlsafe_b64decode(tag + "=")

    assert len(decoded_tag) == hashlib.sha256().digest_size
    assert decoded_tag != hashlib.sha256(b"session_id\0private-value").digest()
    assert (
        hmac.compare_digest(
            decoded_tag,
            hmac.digest(
                base64.urlsafe_b64decode(encoded_key.get_secret_value() + "="),
                # An independently framed payload is intentionally not accepted;
                # the codec owns one exact versioned canonical representation.
                b"private-value",
                "sha256",
            ),
        )
        is False
    )


def test_rotation_signs_with_active_key_and_verifies_retained_keys() -> None:
    first = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="old", keys={"old": _key(1)})
    )
    old_alias = first.encode("session-a", field_name="session_id")

    rotated = first.rotated(active_key_id="new", key=_key(2))
    new_alias = rotated.encode("session-a", field_name="session_id")

    assert rotated.keyring.key_ids == ("new", "old")
    assert ".new.session_id." in new_alias
    assert rotated.matches(old_alias, "session-a", field_name="session_id")
    assert rotated.matches(new_alias, "session-a", field_name="session_id")
    assert not first.matches(new_alias, "session-a", field_name="session_id")

    retired = rotated.rotated(
        active_key_id="new",
        key=_key(2),
        retire_key_ids=("old",),
    )
    assert retired.keyring.key_ids == ("new",)
    assert not retired.matches(old_alias, "session-a", field_name="session_id")
    assert retired.matches(new_alias, "session-a", field_name="session_id")


def test_codec_enumerates_rotation_aliases_and_exposes_safe_key_fingerprints() -> None:
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="new",
            keys={"old": _key(1), "new": _key(2)},
        )
    )

    aliases = codec.aliases("session-a", field_name="session_id")

    assert len(aliases) == 2
    assert ".new.session_id." in aliases[0]
    assert ".old.session_id." in aliases[1]
    assert all(codec.matches(alias, "session-a", field_name="session_id") for alias in aliases)
    assert len(codec.key_fingerprint("new")) == 64
    assert codec.key_fingerprint("new") != codec.key_fingerprint("old")
    assert _key(2).get_secret_value() not in codec.key_fingerprint("new")
    with pytest.raises(ValueError, match="not configured"):
        codec.key_fingerprint("missing")


def test_keyring_is_frozen_secret_safe_and_defensively_copied() -> None:
    raw_key = _key(3)
    source = {"primary": raw_key}
    keyring = PublicAuthorityAliasKeyring(active_key_id="primary", keys=source)
    source["other"] = _key(4)

    assert keyring.key_ids == ("primary",)
    assert isinstance(keyring.keys, Mapping)
    assert keyring.keys["primary"].get_secret_value() == raw_key.get_secret_value()
    assert raw_key.get_secret_value() not in repr(keyring)
    assert raw_key.get_secret_value() not in repr(keyring.keys)
    with pytest.raises(TypeError):
        cast("Any", keyring.keys)["other"] = _key(4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast("Any", keyring).active_key_id = "other"


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {f"key-{index}": _key(index) for index in range(PUBLIC_AUTHORITY_ALIAS_MAX_KEYS + 1)},
    ],
)
def test_keyring_requires_one_to_four_keys(keys: dict[str, SecretStr]) -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        PublicAuthorityAliasKeyring(active_key_id="key-0", keys=keys)


@pytest.mark.parametrize(
    "value",
    [
        SecretStr("a" * 42),
        SecretStr("a" * 44),
        SecretStr(base64.urlsafe_b64encode(bytes(32)).decode()),
        SecretStr("!" * 43),
        SecretStr("A" * 42 + "B"),
    ],
)
def test_keyring_rejects_noncanonical_or_wrong_length_key_material(value: SecretStr) -> None:
    with pytest.raises(ValueError, match="canonical unpadded base64url") as caught:
        PublicAuthorityAliasKeyring(active_key_id="primary", keys={"primary": value})
    assert value.get_secret_value() not in str(caught.value)


def test_keyring_validates_ids_rotation_capacity_and_reassignment() -> None:
    with pytest.raises(ValueError, match="active_key_id"):
        PublicAuthorityAliasKeyring(active_key_id="missing", keys={"primary": _key(1)})
    with pytest.raises(ValueError, match="canonical lowercase"):
        PublicAuthorityAliasKeyring(active_key_id="Primary", keys={"Primary": _key(1)})
    with pytest.raises(TypeError, match="SecretStr"):
        PublicAuthorityAliasKeyring(
            active_key_id="primary",
            keys=cast("Any", {"primary": _key(1).get_secret_value()}),
        )

    keyring = PublicAuthorityAliasKeyring(
        active_key_id="key-0",
        keys={f"key-{index}": _key(index) for index in range(4)},
    )
    with pytest.raises(ValueError, match="between 1 and 4"):
        keyring.rotated(active_key_id="key-4", key=_key(4))
    with pytest.raises(ValueError, match="different key material"):
        keyring.rotated(active_key_id="key-0", key=_key(9))
    with pytest.raises(ValueError, match="active key cannot be retired"):
        keyring.rotated(
            active_key_id="key-4",
            key=_key(4),
            retire_key_ids=("key-4",),
        )


@pytest.mark.parametrize(
    "alias",
    [
        "cayu_authority_session_id_" + "a" * 64,
        "cayu_authority_v2.primary.session_id." + "A" * 43,
        "cayu_authority_v1.Primary.session_id." + "A" * 43,
        "cayu_authority_v1.primary.SessionId." + "A" * 43,
        "cayu_authority_v1.primary.session_id." + "A" * 42,
        "cayu_authority_v1.primary.session_id." + "A" * 43 + "=",
        "cayu_authority_v1.primary.session_id." + "_" * 43,
        "cayu_authority_v1.primary.session_id." + "A" * 43 + ".extra",
    ],
)
def test_parser_rejects_old_unkeyed_and_malformed_aliases_but_reserves_namespace(
    alias: str,
) -> None:
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="primary", keys={"primary": _key(1)})
    )

    assert parse_public_authority_alias(alias) is None
    assert codec.parse(alias) is None
    assert not codec.matches(alias, "private", field_name="session_id")
    assert public_authority_alias_is_reserved(alias)
    assert codec.is_reserved(alias)


def test_parser_and_reserved_namespace_reject_non_strings_without_raising() -> None:
    assert parse_public_authority_alias(cast("Any", None)) is None
    assert not public_authority_alias_is_reserved(None)
    assert not public_authority_alias_is_reserved(1)
    assert not public_authority_alias_is_reserved("ordinary-session")


def test_codec_accepts_unicode_authority_but_rejects_invalid_inputs() -> None:
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="primary", keys={"primary": _key(1)})
    )
    alias = codec.encode(
        "interaction-🧭",
        field_name="interaction_id",
        session_id="session-世界",
    )
    assert codec.matches(
        alias,
        "interaction-🧭",
        field_name="interaction_id",
        session_id="session-世界",
    )

    with pytest.raises(ValueError, match="non-empty"):
        codec.encode("", field_name="session_id")
    with pytest.raises(ValueError, match="canonical lowercase"):
        codec.encode("value", field_name="SessionId")
    with pytest.raises(ValueError, match="surrogate"):
        codec.encode("\ud800", field_name="session_id")


def test_deployment_codec_loader_requires_complete_explicit_environment() -> None:
    assert public_authority_alias_codec_from_environment({}) is None
    with pytest.raises(ValueError, match="requires both"):
        public_authority_alias_codec_from_environment(
            {"CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID": "primary"}
        )

    codec = public_authority_alias_codec_from_environment(
        {
            "CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID": "primary",
            "CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS": (
                '{"primary":"' + _key(1).get_secret_value() + '"}'
            ),
        }
    )

    assert codec is not None
    alias = codec.encode("session", field_name="session_id")
    assert codec.matches(alias, "session", field_name="session_id")
