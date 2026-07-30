from __future__ import annotations

import asyncio
import time
from itertools import product

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    REDACTED_SECRET,
    ChainVault,
    Environment,
    EnvironmentSpec,
    LocalEnvVault,
    PassthroughProxy,
    ResolvedSecret,
    RoutedVault,
    SecretEnv,
    SecretNotFound,
    SecretRedactor,
    SecretRef,
    StaticVault,
    Vault,
    VaultError,
    copy_resolved_secret,
    copy_secret_env,
    resolve_secret_env,
    secret_env_refs,
    validate_secret_resolver,
)
from cayu.vaults import SecretRedactionTail
from cayu.vaults.redaction import (
    SecretRedactionCapacityError,
    _source_redaction_pieces,
)


def test_secret_env_is_reference_only_and_owns_metadata() -> None:
    metadata = {"scope": {"project": "alpha"}}
    ref = SecretRef(name="github_token", handle="vault://github", metadata=metadata)
    secret_env = SecretEnv(name="GITHUB_TOKEN", ref=ref, metadata=metadata)

    metadata["scope"]["project"] = "mutated"
    ref.metadata["scope"]["project"] = "mutated-ref"
    dumped = secret_env.model_dump()

    assert dumped == {
        "name": "GITHUB_TOKEN",
        "ref": {
            "name": "github_token",
            "handle": "vault://github",
            "metadata": {"scope": {"project": "alpha"}},
        },
        "metadata": {"scope": {"project": "alpha"}},
    }
    assert "value" not in dumped


def test_secret_env_rejects_invalid_boundary_data() -> None:
    with pytest.raises(ValidationError, match="cannot be blank"):
        SecretEnv(name=" ", ref=SecretRef(name="github_token"))

    with pytest.raises(ValidationError, match="extra"):
        SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token"), value="secret")  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="JSON-compatible"):
        SecretEnv(
            name="GITHUB_TOKEN",
            ref=SecretRef(name="github_token"),
            metadata={"bad": object()},
        )


def test_copy_secret_env_rejects_subclasses_before_attribute_access() -> None:
    class BadSecretEnv(SecretEnv):
        def __getattribute__(self, name):
            if name == "name":
                raise RuntimeError("secret env name access should not run")
            return super().__getattribute__(name)

    secret_env = BadSecretEnv.model_construct(
        name="TOKEN",
        ref=SecretRef(name="token"),
        metadata={},
    )

    with pytest.raises(TypeError, match="SecretEnv"):
        copy_secret_env(secret_env)


def test_copy_resolved_secret_owns_value_and_metadata() -> None:
    metadata = {"scope": {"project": "alpha"}}
    secret = ResolvedSecret(
        name="github_token",
        value=SecretStr("ghp_secret"),
        metadata=metadata,
    )

    copied = copy_resolved_secret(secret)
    secret.value = SecretStr("mutated_secret")
    secret.metadata["scope"]["project"] = "mutated"
    metadata["scope"]["project"] = "external-mutated"

    assert copied == ResolvedSecret(
        name="github_token",
        value=SecretStr("ghp_secret"),
        metadata={"scope": {"project": "alpha"}},
    )


def test_copy_resolved_secret_rejects_subclasses_before_attribute_access() -> None:
    class BadResolvedSecret(ResolvedSecret):
        def __getattribute__(self, name):
            if name == "name":
                raise RuntimeError("secret name access should not run")
            return super().__getattribute__(name)

    secret = BadResolvedSecret.model_construct(
        name="token",
        value=SecretStr("secret"),
        metadata={},
    )

    with pytest.raises(TypeError, match="ResolvedSecret"):
        copy_resolved_secret(secret)


def test_paginated_redaction_batches_a_large_unmatched_source_run() -> None:
    redactor = SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP")
    source = b"x" * (4 * 1024 * 1024)

    pieces = _source_redaction_pieces(
        source,
        ordered_patterns=redactor._ordered_byte_patterns(),
    )

    assert len(pieces) == 1
    assert pieces[0].value == source
    assert pieces[0].source_start == 0
    assert pieces[0].source_end == len(source)
    assert pieces[0].linear is True


def test_static_vault_gets_and_resolves_secret_refs() -> None:
    vault = StaticVault(
        {"github_token": "ghp_test"},
        metadata={"github_token": {"owner": "user_1"}},
    )

    ref = asyncio.run(vault.get("github_token", scope={"session_id": "sess_1"}))
    resolved = asyncio.run(vault.resolve(ref, scope={"session_id": "sess_1"}))

    assert ref == SecretRef(
        name="github_token",
        handle="static:github_token",
        metadata={"owner": "user_1", "scope": {"session_id": "sess_1"}},
    )
    assert resolved.name == "github_token"
    assert str(resolved.value) == "**********"
    assert resolved.value.get_secret_value() == "ghp_test"
    assert resolved.metadata == {"owner": "user_1", "scope": {"session_id": "sess_1"}}


def test_static_vault_rejects_missing_and_blank_secrets() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        StaticVault({"github_token": " "})

    with pytest.raises(ValueError, match="cannot be blank"):
        StaticVault({"github_token": SecretStr(" ")})

    vault = StaticVault({"github_token": "ghp_test"})
    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.get("missing"))

    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.resolve(SecretRef(name="missing")))


def test_local_env_vault_resolves_trusted_process_env(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_TEST_GITHUB_TOKEN", "ghp_from_env")
    vault = LocalEnvVault(
        {"github_token": "CAYU_TEST_GITHUB_TOKEN"},
        metadata={"github_token": {"source": "env"}},
    )

    ref = asyncio.run(vault.get("github_token"))
    resolved = asyncio.run(vault.resolve(ref))

    assert ref == SecretRef(
        name="github_token",
        handle="env:CAYU_TEST_GITHUB_TOKEN",
        metadata={"source": "env"},
    )
    assert resolved.value.get_secret_value() == "ghp_from_env"
    assert resolved.metadata == {"source": "env"}


def test_local_env_vault_rejects_missing_mapping_and_unset_env(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_TEST_MISSING_TOKEN", raising=False)
    monkeypatch.setenv("CAYU_TEST_BLANK_TOKEN", " ")
    vault = LocalEnvVault({"github_token": "CAYU_TEST_MISSING_TOKEN"})

    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.get("missing"))

    with pytest.raises(SecretNotFound, match="not set"):
        asyncio.run(vault.resolve(SecretRef(name="github_token")))

    blank_vault = LocalEnvVault({"github_token": "CAYU_TEST_BLANK_TOKEN"})
    with pytest.raises(SecretNotFound, match="blank"):
        asyncio.run(blank_vault.resolve(SecretRef(name="github_token")))


def test_environment_resolves_secret_through_attached_vault() -> None:
    environment = Environment(
        EnvironmentSpec(name="local"),
        vault=StaticVault({"github_token": "ghp_test"}),
    )

    resolved = asyncio.run(environment.resolve_secret(SecretRef(name="github_token")))

    assert resolved.value.get_secret_value() == "ghp_test"


def test_environment_requires_vault_for_secret_resolution() -> None:
    environment = Environment(EnvironmentSpec(name="local"))

    with pytest.raises(VaultError, match="no vault"):
        asyncio.run(environment.resolve_secret(SecretRef(name="github_token")))


def test_secret_redactor_redacts_strings_and_json_keys_and_values() -> None:
    resolved = ResolvedSecret(name="github_token", value=SecretStr("ghp_secret"))
    redactor = SecretRedactor([resolved]).with_secret("npm_secret")

    assert redactor.redact_text("tokens: ghp_secret npm_secret") == (
        f"tokens: {REDACTED_SECRET} {REDACTED_SECRET}"
    )
    assert redactor.redact_json(
        {
            "stdout": "ghp_secret",
            "nested": ["npm_secret", {"safe": "ok", "npm_secret-key": "value"}],
        }
    ) == {
        "stdout": REDACTED_SECRET,
        "nested": [
            REDACTED_SECRET,
            {"safe": "ok", f"{REDACTED_SECRET}-key": "value"},
        ],
    }


def test_secret_redactor_accepts_single_secret_values() -> None:
    assert SecretRedactor("token").redact_text("token total") == (f"{REDACTED_SECRET} total")
    assert SecretRedactor(SecretStr("token")).redact_text("token total") == (
        f"{REDACTED_SECRET} total"
    )
    assert (
        SecretRedactor(ResolvedSecret(name="github_token", value=SecretStr("token"))).redact_text(
            "token total"
        )
        == f"{REDACTED_SECRET} total"
    )


def test_secret_redactor_preserves_values_when_redacted_keys_collide() -> None:
    redactor = SecretRedactor("secret")

    redacted = redactor.redact_json(
        {
            "secret": {"value": 2},
            REDACTED_SECRET: {"value": 1},
        }
    )

    assert redacted == {
        f"{REDACTED_SECRET}_2": {"value": 2},
        REDACTED_SECRET: {"value": 1},
    }


def test_secret_redactor_exposes_whether_it_has_values() -> None:
    assert SecretRedactor().has_values is False
    assert SecretRedactor("token").has_values is True


def test_secret_redactor_redacts_json_values_without_rewriting_protocol_fields() -> None:
    redactor = SecretRedactor(["tool", "deny", "secret"])

    redacted = redactor.redact_json_values(
        {
            "tool_name": "tool",
            "decision": "deny",
            "arguments": {"secret-key": "secret"},
        },
        preserve_string_fields={"tool_name", "decision"},
    )

    assert redacted == {
        "tool_name": "tool",
        "decision": "deny",
        "arguments": {"secret-key": REDACTED_SECRET},
    }


@pytest.mark.parametrize("preserved_key", ["error", "result", "metadata", "tool_name"])
def test_secret_redactor_rejects_preserved_key_spellings_inside_untrusted_data(
    preserved_key: str,
) -> None:
    redactor = SecretRedactor(preserved_key)

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redactor.require_no_secret_keys(
            {
                "metadata": {
                    preserved_key: "caller-controlled",
                }
            },
            preserve_keys={"metadata", "error", "result", "tool_name"},
            untrusted_container_keys={"metadata"},
        )

    # The same spellings remain valid at their documented structural boundary.
    redactor.require_no_secret_keys(
        {preserved_key: "typed structure"},
        preserve_keys={"metadata", "error", "result", "tool_name"},
        untrusted_container_keys={"metadata"},
    )


def test_secret_redactor_is_idempotent_when_a_secret_overlaps_the_marker() -> None:
    marker_prefixed_secret = f"{REDACTED_SECRET}-credential"
    redactor = SecretRedactor(["REDA", "secret", marker_prefixed_secret])
    once = redactor.redact_text(f"prefix secret {marker_prefixed_secret} suffix")

    assert once == f"prefix {REDACTED_SECRET} {REDACTED_SECRET} suffix"
    assert redactor.redact_text(once) == once


def test_secret_redactor_does_not_mask_diagnostics_containing_lone_surrogates() -> None:
    redactor = SecretRedactor(["secret", "\ud800token"])

    redacted = redactor.redact_text("\udcff secret \ud800token \udfff")

    assert redacted == f"\udcff {REDACTED_SECRET} {REDACTED_SECRET} \udfff"
    assert redactor.redact_text(redacted) == redacted


@pytest.mark.parametrize(
    ("secrets", "source"),
    [
        (["b密[", "c[[]b"], "b密c[[]b"),
        (["]["], "]][["),
        (["]tail", "head["], "head[secret]tail"),
        ([f"{REDACTED_SECRET}-credential"], f"{REDACTED_SECRET}-credential"),
    ],
)
def test_secret_redactor_collapses_secrets_reconstructed_across_marker_edges(
    secrets: list[str],
    source: str,
) -> None:
    redactor = SecretRedactor(secrets)

    once = redactor.redact_text(source)

    assert redactor.redact_text(once) == once
    assert not any(secret in once for secret in secrets)


def test_secret_redactor_reports_longest_utf8_secret_for_bounded_overlap() -> None:
    redactor = SecretRedactor(["ascii", "密钥值"])

    assert redactor.max_secret_utf8_bytes == len("密钥值".encode())
    assert SecretRedactor().max_secret_utf8_bytes == 0


def test_secret_redactor_discards_ambiguous_pretruncated_prefix_at_every_split() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    redactor = SecretRedactor(secret)
    source = f"safe:{secret}".encode()

    for split in range(len(b"safe:") + 1, len(source)):
        projected, truncated = redactor.redact_utf8_head(
            source[:split],
            max_bytes=split,
            source_complete=False,
        )

        assert projected == "safe:"
        assert truncated is True


@pytest.mark.parametrize("max_bytes", [1, 16, len(REDACTED_SECRET), 64])
def test_secret_redactor_bounds_complete_text_only_after_redaction(max_bytes: int) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    redactor = SecretRedactor(secret)

    projected, truncated = redactor.redact_text_head(secret, max_bytes=max_bytes)

    assert secret not in projected
    assert secret[:16] not in projected
    assert len(projected.encode()) <= max_bytes
    assert truncated is (len(REDACTED_SECRET.encode()) > max_bytes)


def test_paginated_redaction_cannot_reconstruct_secret_across_visible_pages() -> None:
    redactor = SecretRedactor("CE")
    source = b"CCEE"
    visible_pages: list[str] = []

    for page_offset in range(0, len(source), 2):
        page_end = min(page_offset + 2, len(source))
        text, _ = redactor.redact_utf8_page(
            source,
            window_offset=0,
            page_offset=page_offset,
            page_end=page_end,
            max_bytes=2,
            source_complete=page_end == len(source),
        )
        visible_pages.append(text)

    assert visible_pages == ["", "E"]
    assert "CE" not in "".join(visible_pages)


@pytest.mark.parametrize(
    "secrets",
    [
        ["stream-boundary-secret"],
        ["密钥值"],
        ["a"],
        ["abc", "bcde", "abcdef"],
        ["repeat-secret-value", f"{REDACTED_SECRET}-credential", "REDA"],
    ],
)
def test_secret_redaction_stream_matches_whole_value_at_every_byte_split(
    secrets: list[str],
) -> None:
    redactor = SecretRedactor(secrets)
    source = (f"prefix {secrets[0]} middle {secrets[-1]} {REDACTED_SECRET} suffix").encode()
    expected = redactor.redact_text(source.decode()).encode()

    for split in range(len(source) + 1):
        stream = redactor.stream_bytes()
        actual = (
            stream.feed(source[:split]) + stream.feed(source[split:]) + stream.finish_complete()
        )
        assert actual == expected


@pytest.mark.parametrize(
    ("secrets", "source"),
    [
        (["b密[", "c[[]b"], "b密c[[]bcéab"),
        (["]["], "]][["),
        (["]tail", "head["], "head[secret]tail"),
        (["b_", "密", "]["], "[cb_[]密b]"),
        (["[", "]密", "密["], "[[密密[密é]ba]密][b"),
        (["[", "aaaaaaaa["], "aaaaaaaaaaaaaaaaaaaaaaaa["),
        (["[", "]]["], "[]]]]]["),
        (
            [f"{REDACTED_SECRET}a", f"{REDACTED_SECRET}aSECRET]"],
            f"{REDACTED_SECRET}aaaSECRET]",
        ),
        (
            ["][", "SECRET][R[R", "a", "密b"],
            (f"密{REDACTED_SECRET}a[[R{REDACTED_SECRET}abSECRET][[ba{REDACTED_SECRET}b"),
        ),
    ],
)
def test_secret_redaction_stream_is_fixed_point_at_every_split(
    secrets: list[str],
    source: str,
) -> None:
    redactor = SecretRedactor(secrets)
    encoded = source.encode("utf-8")
    expected = redactor.redact_text(source)

    for split in range(len(encoded) + 1):
        stream = redactor.stream_bytes()
        output = (
            stream.feed(encoded[:split]) + stream.feed(encoded[split:]) + stream.finish_complete()
        ).decode("utf-8")

        assert output == expected
        assert redactor.redact_text(output) == output


@pytest.mark.parametrize(
    "secrets",
    [
        ["[", "]密", "密["],
        ["b_", "密", "]["],
        ["[a", "a]", "]["],
    ],
)
def test_secret_redaction_stream_matches_whole_value_for_exhaustive_short_sources(
    secrets: list[str],
) -> None:
    redactor = SecretRedactor(secrets)

    for length in range(6):
        for characters in product("[]a密", repeat=length):
            source = "".join(characters)
            encoded = source.encode()
            expected = redactor.redact_text(source).encode()
            stream = redactor.stream_bytes()
            bytewise = b"".join(stream.feed(bytes([byte])) for byte in encoded)
            bytewise += stream.finish_complete()

            assert bytewise == expected
            assert redactor.redact_text(bytewise.decode()) == bytewise.decode()


def test_secret_redaction_stream_abort_discards_provisional_marker_merge() -> None:
    redactor = SecretRedactor(["[", "]密", "密["])
    source = "[[密密[密é]ba]密][b".encode()
    stream = redactor.stream_bytes()

    released = stream.feed(source[:7])
    discarded = stream.abort()

    assert discarded is True
    assert redactor.redact_text(released.decode()) == released.decode()
    assert redactor.redact_text(source.decode()).encode().startswith(released)
    assert stream.feed(source[7:]) == b""
    assert stream.finish_complete() == b""


def test_secret_redactor_handles_marker_heavy_short_secret_in_bounded_time() -> None:
    redactor = SecretRedactor("[")
    source = "[" * 10_000

    started = time.perf_counter()
    redacted = redactor.redact_text(source)
    elapsed = time.perf_counter() - started

    assert redacted == REDACTED_SECRET * len(source)
    assert redactor.redact_text(redacted) == redacted
    assert elapsed < 3


def test_secret_redaction_stream_compacts_unresolved_marker_runs() -> None:
    redactor = SecretRedactor(["[", "]["])
    stream = redactor.stream_bytes()
    marker_bytes = REDACTED_SECRET.encode()

    started = time.perf_counter()
    released = bytearray()
    max_storage_bytes = 0
    for _ in range(10_000):
        released.extend(stream.feed(b"["))
        max_storage_bytes = max(
            max_storage_bytes,
            stream._redacted_pending.storage_bytes,
        )
    released.extend(stream.finish_complete())
    elapsed = time.perf_counter() - started

    assert bytes(released) == marker_bytes
    assert max_storage_bytes == len(marker_bytes)
    assert elapsed < 3


def test_secret_redaction_stream_compacts_homogeneous_fenced_literals() -> None:
    stream = SecretRedactor(["[", "]a"]).stream_bytes()

    assert stream.feed(b"[" + b"a" * 100_000) == b""

    assert stream._redacted_pending.storage_bytes == len(REDACTED_SECRET.encode()) + 1
    assert stream.finish_complete() == REDACTED_SECRET.encode()


def test_secret_redaction_stream_fails_closed_before_bounded_state_expands() -> None:
    stream = SecretRedactor(["[", "]a"]).stream_bytes(max_retained_bytes=1024)

    with pytest.raises(
        SecretRedactionCapacityError,
        match="unresolved-source capacity",
    ):
        stream.feed(b"[" + b"a" * 100_000)

    assert stream._redacted_pending.storage_bytes == 0
    assert stream.feed(b"late") == b""
    assert stream.finish_complete() == b""


def test_secret_redaction_capacity_error_carries_only_the_proven_prefix() -> None:
    stream = SecretRedactor(["[", "]a"]).stream_bytes(max_retained_bytes=1024)

    with pytest.raises(SecretRedactionCapacityError) as raised:
        stream.feed(b"public:" + b"[" + b"a" * 10_000)

    assert raised.value.released == b"public:"
    assert stream.feed(b"late") == b""


def test_secret_redaction_stream_rechecks_when_literal_resolves_marker_run() -> None:
    stream = SecretRedactor(["[", "]["]).stream_bytes()

    assert stream.feed(b"[") == b""
    assert stream.feed(b"[") == b""
    assert stream.feed(b"[") == b""
    assert stream.feed(b"x") == f"{REDACTED_SECRET}x".encode()
    assert stream.abort() is False


def test_literal_marker_byte_is_a_hard_barrier_when_no_secret_uses_it() -> None:
    stream = SecretRedactor(["[", "]["]).stream_bytes()

    assert stream.feed(b"[") == b""
    assert stream.feed(b"RR") == f"{REDACTED_SECRET}RR".encode()
    assert stream.abort() is False


def test_secret_redaction_stream_discards_an_undecided_suffix_when_aborted() -> None:
    secret = "abandonment-secret"
    stream = SecretRedactor(secret).stream_bytes()

    released = stream.feed(f"safe {secret[:-1]}".encode())
    stream.abort()

    assert released == b"safe "
    assert stream.feed(b"late-callback-data") == b""
    assert stream.finish_complete() == b""


def test_secret_redaction_stream_releases_an_unmatched_suffix_only_at_proven_eof() -> None:
    stream = SecretRedactor("secret-value").stream_bytes()

    released = stream.feed(b"safe sec")

    assert released == b"safe "
    assert stream.finish_complete() == b"sec"


def test_secret_redaction_tail_matches_before_reverse_bound_at_every_byte_split() -> None:
    secret = "rolling-tail-boundary-secret"
    max_bytes = 64
    source = (f"{'x' * 90}{secret}{'z' * 48}").encode()

    for split in range(len(source) + 1):
        tail = SecretRedactionTail(SecretRedactor(secret), max_bytes=max_bytes)
        tail.feed(source[:split])
        tail.feed(source[split:])
        tail.finish_complete()
        result = tail.text()

        assert len(result.encode()) <= max_bytes
        assert secret not in result
        assert secret[:12] not in result
        assert secret[-12:] not in result
        assert REDACTED_SECRET in result


def test_secret_redaction_tail_is_split_independent_for_marker_overlap_chain() -> None:
    secrets = ["[", "]密", "密["]
    source = ("public-prefix:" + "[[密密[密é]ba]密][b" + ":public-suffix").encode()
    redactor = SecretRedactor(secrets)
    expected_tail = SecretRedactionTail(redactor, max_bytes=64)
    expected_tail.feed(source)
    expected_tail.finish_complete()
    expected = expected_tail.text()

    for split in range(len(source) + 1):
        tail = SecretRedactionTail(redactor, max_bytes=64)
        tail.feed(source[:split])
        tail.feed(source[split:])
        tail.finish_complete()

        assert tail.text() == expected
        assert redactor.redact_text(tail.text()) == tail.text()


def test_secret_redaction_tail_respects_a_marker_sized_bound() -> None:
    secret = "tail-secret"
    max_bytes = len(REDACTED_SECRET.encode())
    tail = SecretRedactionTail(SecretRedactor(secret), max_bytes=max_bytes)

    tail.feed(f"prefix {secret} suffix".encode())
    tail.finish_complete()

    assert tail.text() == REDACTED_SECRET
    assert len(tail.text().encode()) == max_bytes


def test_secret_redaction_tail_never_retains_a_partial_marker_from_a_run() -> None:
    tail = SecretRedactionTail(
        SecretRedactor("["),
        max_bytes=2 * len(REDACTED_SECRET.encode()) - 2,
    )

    tail.feed(b"[" * 10_000)
    tail.finish_complete()

    assert tail.text() == REDACTED_SECRET


def test_secret_redaction_tail_discards_uncertain_suffix_when_drain_is_cancelled() -> None:
    secret = "rolling-tail-abandonment-secret"
    tail = SecretRedactionTail(SecretRedactor(secret), max_bytes=64)
    tail.feed(f"{'x' * 90}:{secret[:14]}".encode())

    tail.abort()
    tail.feed(secret[14:].encode())
    result = tail.text()

    assert len(result.encode()) <= 64
    assert secret[:14] not in result
    assert result.endswith(":")


def test_secret_redaction_tail_omits_an_over_capacity_ambiguous_chain() -> None:
    tail = SecretRedactionTail(
        SecretRedactor(["[", "]a"]),
        max_bytes=32,
    )

    tail.feed(b"public:" + b"[" + b"a" * 100_000)
    result = tail.text()

    assert result == "public:"
    assert "[" not in result


def test_secret_redaction_tail_accepts_determined_output_at_eof() -> None:
    tail = SecretRedactionTail(
        SecretRedactor(["a", "a" * 10_000]),
        max_bytes=32,
    )

    tail.feed(b"a" * 9_999)
    tail.finish_complete()

    assert tail.text() == REDACTED_SECRET


def test_secret_redaction_tail_matches_secrets_synthesized_by_utf8_replacement() -> None:
    secret = "a\ufffdb"
    tail = SecretRedactionTail(SecretRedactor(secret), max_bytes=64)

    tail.feed(b"prefix:a")
    tail.feed(b"\xff")
    tail.feed(b"b:suffix")
    tail.finish_complete()

    assert tail.text() == f"prefix:{REDACTED_SECRET}:suffix"


@pytest.mark.parametrize("max_bytes", range(1, len(REDACTED_SECRET.encode()) + 3))
def test_bounded_redaction_never_splits_its_public_marker(max_bytes: int) -> None:
    redactor = SecretRedactor(["secret", "REDA"])
    bounded = redactor.redact_text_bounded(
        "prefix secret suffix",
        max_bytes=max_bytes,
    )

    encoded = bounded.encode()
    assert len(encoded) <= max_bytes
    assert redactor.redact_text(bounded) == bounded
    assert bounded not in {REDACTED_SECRET[:length] for length in range(1, len(REDACTED_SECRET))}


def test_bounded_redaction_is_stable_after_repeated_matches_shift_a_later_boundary() -> None:
    secret = "long-repeated-secret-value"
    redactor = SecretRedactor([secret, "REDA"])
    source = f"{secret}{'x' * 20}{secret}"

    bounded = redactor.redact_text_bounded(source, max_bytes=50)

    assert len(bounded.encode()) <= 50
    assert secret not in bounded
    assert not bounded.endswith(tuple(secret[:index] for index in range(1, len(secret))))
    assert redactor.redact_text_bounded(bounded, max_bytes=50) == bounded


def test_secret_redactor_rejects_non_json_values() -> None:
    redactor = SecretRedactor(["secret"])

    with pytest.raises(ValueError, match="JSON-compatible"):
        redactor.redact_json(object())


def test_secret_redactor_rejects_blank_secrets() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor([" "])

    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor().with_secret(SecretStr(" "))


class _StubVault(Vault):
    """Configurable vault: records calls, optionally raises a non-SecretNotFound error."""

    def __init__(
        self, secrets: dict[str, str] | None = None, *, error: Exception | None = None
    ) -> None:
        self._secrets = dict(secrets or {})
        self._error = error
        self.get_calls: list[tuple[str, dict | None]] = []
        self.resolve_calls: list[tuple[str, dict | None]] = []

    async def get(self, name: str, *, scope: dict | None = None) -> SecretRef:
        self.get_calls.append((name, scope))
        if self._error is not None:
            raise self._error
        if name not in self._secrets:
            raise SecretNotFound(f"Secret not found: {name}")
        return SecretRef(name=name, handle=f"stub:{name}")

    async def resolve(self, ref: SecretRef, *, scope: dict | None = None) -> ResolvedSecret:
        self.resolve_calls.append((ref.name, scope))
        if self._error is not None:
            raise self._error
        if ref.name not in self._secrets:
            raise SecretNotFound(f"Secret not found: {ref.name}")
        return ResolvedSecret(name=ref.name, value=SecretStr(self._secrets[ref.name]))


# --- ChainVault -----------------------------------------------------------------------


def test_chain_vault_first_success_wins_and_short_circuits() -> None:
    first = _StubVault({"token": "one"})
    second = _StubVault({"token": "two"})
    chain = ChainVault(first, second)

    ref = asyncio.run(chain.get("token"))
    resolved = asyncio.run(chain.resolve(ref))

    assert resolved.value.get_secret_value() == "one"
    assert second.get_calls == [] and second.resolve_calls == []  # short-circuited


def test_chain_vault_falls_through_to_next_on_secret_not_found() -> None:
    first = _StubVault({})  # knows nothing
    second = _StubVault({"token": "two"})
    chain = ChainVault(first, second)

    resolved = asyncio.run(chain.resolve(SecretRef(name="token")))

    assert resolved.value.get_secret_value() == "two"
    assert first.resolve_calls == [("token", None)]  # was tried first


def test_chain_vault_raises_when_no_vault_resolves() -> None:
    chain = ChainVault(_StubVault({}), _StubVault({}))
    with pytest.raises(SecretNotFound, match="No vault could resolve"):
        asyncio.run(chain.get("missing"))
    with pytest.raises(SecretNotFound, match="No vault could resolve"):
        asyncio.run(chain.resolve(SecretRef(name="missing")))


def test_chain_vault_propagates_non_secret_not_found_errors() -> None:
    # A real failure (e.g. a network error) must NOT be swallowed by a later vault.
    failing = _StubVault({}, error=VaultError("nango down"))
    backup = _StubVault({"token": "two"})
    chain = ChainVault(failing, backup)

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(chain.resolve(SecretRef(name="token")))
    assert backup.resolve_calls == []  # not reached — the error stopped the chain

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(chain.get("token"))
    assert backup.get_calls == []  # get honors the same propagation contract


def test_chain_vault_passes_scope_through() -> None:
    stub = _StubVault({"token": "one"})
    chain = ChainVault(stub)
    asyncio.run(chain.get("token", scope={"tenant": "t1"}))
    assert stub.get_calls == [("token", {"tenant": "t1"})]


def test_chain_vault_validates_construction() -> None:
    with pytest.raises(ValueError, match="at least one vault"):
        ChainVault()
    with pytest.raises(TypeError, match="Vault instances"):
        ChainVault("not-a-vault")  # type: ignore[arg-type]


# --- RoutedVault ----------------------------------------------------------------------


def test_routed_vault_routes_by_name_with_fallback() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    static = _StubVault({"openai_key": "sk-static"})
    routed = RoutedVault(routes={"gmail": dynamic}, fallback=static)

    routed_result = asyncio.run(routed.resolve(SecretRef(name="gmail")))
    fallback_result = asyncio.run(routed.resolve(SecretRef(name="openai_key")))

    assert routed_result.value.get_secret_value() == "oauth"
    assert fallback_result.value.get_secret_value() == "sk-static"
    assert dynamic.resolve_calls == [("gmail", None)]  # static never hit for gmail
    assert static.resolve_calls == [("openai_key", None)]


def test_routed_vault_get_routes_by_name() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    static = _StubVault({"openai_key": "sk-static"})
    routed = RoutedVault(routes={"gmail": dynamic}, fallback=static)

    ref = asyncio.run(routed.get("gmail"))
    assert ref.handle == "stub:gmail"
    assert static.get_calls == []


def test_routed_vault_unrouted_without_fallback_raises_without_calling_vaults() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    routed = RoutedVault(routes={"gmail": dynamic})
    with pytest.raises(SecretNotFound, match="No vault configured"):
        asyncio.run(routed.get("slack"))
    assert dynamic.get_calls == []  # never called for an unrouted name


def test_routed_vault_passes_scope_through() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    routed = RoutedVault(routes={"gmail": dynamic})
    asyncio.run(routed.resolve(SecretRef(name="gmail"), scope={"connection_id": "org_123"}))
    assert dynamic.resolve_calls == [("gmail", {"connection_id": "org_123"})]


def test_routed_vault_validates_construction() -> None:
    with pytest.raises(ValueError, match="route or a fallback"):
        RoutedVault({})
    with pytest.raises(TypeError, match="must be a mapping"):
        RoutedVault(["gmail"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be blank"):
        RoutedVault({" ": _StubVault({})})
    with pytest.raises(TypeError, match="Vault instances"):
        RoutedVault({"gmail": "nope"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="fallback must be a Vault"):
        RoutedVault({}, fallback="nope")  # type: ignore[arg-type]


def test_composites_work_with_real_vaults() -> None:
    # End-to-end against real StaticVault/LocalEnvVault, exercising the real SecretNotFound
    # fall-through, and usable as an Environment's single vault.
    static = StaticVault({"openai_key": "sk-real"})
    other = StaticVault({"anthropic_key": "ak-real"})

    chained = ChainVault(other, static)
    routed = RoutedVault(routes={"openai_key": static}, fallback=other)

    assert (
        asyncio.run(
            chained.resolve(asyncio.run(chained.get("openai_key")))
        ).value.get_secret_value()
        == "sk-real"
    )
    assert (
        asyncio.run(
            routed.resolve(asyncio.run(routed.get("anthropic_key")))
        ).value.get_secret_value()
        == "ak-real"
    )
    # composites are a drop-in Vault for an Environment
    Environment(EnvironmentSpec(name="prod"), vault=chained)
    Environment(EnvironmentSpec(name="prod"), vault=routed)


def test_chain_vault_validates_lookup_inputs() -> None:
    # ChainVault honors the same input contract as the rest of the vault family.
    chain = ChainVault(_StubVault({"token": "one"}))
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(chain.get(" "))
    with pytest.raises(TypeError, match="SecretRef instances"):
        asyncio.run(chain.resolve("not-a-ref"))  # type: ignore[arg-type]


def test_routed_vault_validates_lookup_inputs() -> None:
    # RoutedVault honors the same input contract as the rest of the vault family.
    routed = RoutedVault(routes={"token": _StubVault({"token": "one"})})
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(routed.get(" "))
    with pytest.raises(TypeError, match="SecretRef instances"):
        asyncio.run(routed.resolve("not-a-ref"))  # type: ignore[arg-type]


def test_routed_vault_route_wins_over_fallback() -> None:
    # A name present in BOTH a route and the fallback resolves via the route; fallback untouched.
    routed_vault = _StubVault({"token": "routed"})
    fallback = _StubVault({"token": "fallback"})
    routed = RoutedVault(routes={"token": routed_vault}, fallback=fallback)

    resolved = asyncio.run(routed.resolve(SecretRef(name="token")))

    assert resolved.value.get_secret_value() == "routed"
    assert fallback.resolve_calls == []


def test_routed_vault_propagates_non_secret_not_found_errors() -> None:
    # A routed vault's real error propagates; it does NOT fall through to the fallback.
    failing = _StubVault({}, error=VaultError("nango down"))
    fallback = _StubVault({"gmail": "static"})
    routed = RoutedVault(routes={"gmail": failing}, fallback=fallback)

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(routed.resolve(SecretRef(name="gmail")))
    assert fallback.resolve_calls == []


def test_composites_nest() -> None:
    # Composites are themselves Vaults, so they nest: a miss in the inner composite falls
    # through to the outer chain.
    inner = RoutedVault(routes={"gmail": _StubVault({"gmail": "oauth"})})
    outer = ChainVault(inner, _StubVault({"openai_key": "sk-static"}))

    gmail = asyncio.run(outer.resolve(SecretRef(name="gmail")))
    openai = asyncio.run(outer.resolve(SecretRef(name="openai_key")))

    assert gmail.value.get_secret_value() == "oauth"
    assert openai.value.get_secret_value() == "sk-static"


def test_resolve_secret_env_resolves_entries_and_mappings() -> None:
    vault = StaticVault({"github_token": "gh-secret", "db_password": "db-secret"})
    entries = [
        SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token")),
        SecretEnv(name="DB_PASSWORD", ref=SecretRef(name="db_password")),
    ]

    from_entries = asyncio.run(resolve_secret_env(entries, vault))
    from_mapping = asyncio.run(
        resolve_secret_env({"GITHUB_TOKEN": SecretRef(name="github_token")}, vault)
    )

    assert from_entries["GITHUB_TOKEN"].value.get_secret_value() == "gh-secret"
    assert from_entries["DB_PASSWORD"].value.get_secret_value() == "db-secret"
    assert from_mapping["GITHUB_TOKEN"].value.get_secret_value() == "gh-secret"


def test_resolve_secret_env_passes_scope_and_supports_proxies() -> None:
    proxy = PassthroughProxy(StaticVault({"github_token": "gh-secret"}))

    resolved = asyncio.run(
        resolve_secret_env(
            [SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token"))],
            proxy,
            scope={"session_id": "sess_1"},
        )
    )

    assert resolved["GITHUB_TOKEN"].metadata["scope"] == {"session_id": "sess_1"}


def test_secret_env_refs_rejects_duplicates_and_bad_shapes() -> None:
    ref = SecretRef(name="github_token")

    with pytest.raises(ValueError, match="duplicate"):
        secret_env_refs(
            [
                SecretEnv(name="GITHUB_TOKEN", ref=ref),
                SecretEnv(name="GITHUB_TOKEN", ref=ref),
            ]
        )

    with pytest.raises(TypeError, match="SecretEnv"):
        secret_env_refs(["not-a-secret-env"])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="secret_env"):
        secret_env_refs("GITHUB_TOKEN")  # type: ignore[arg-type]


def test_validate_secret_resolver_requires_async_resolve() -> None:
    class SyncResolver:
        def resolve(self, ref, *, scope=None):
            return None

    validate_secret_resolver(StaticVault({"token": "x"}))
    validate_secret_resolver(PassthroughProxy(StaticVault({"token": "x"})))

    with pytest.raises(TypeError, match="async"):
        validate_secret_resolver(SyncResolver())

    with pytest.raises(TypeError, match="resolve"):
        validate_secret_resolver(object())

    with pytest.raises(TypeError, match="resolve"):
        asyncio.run(
            resolve_secret_env({"GITHUB_TOKEN": SecretRef(name="token")}, object())  # type: ignore[arg-type]
        )


def test_binary_page_redaction_tracks_source_positions_without_fragment_leak() -> None:
    secret = "binary-page-secret-canary"
    redactor = SecretRedactor(secret)
    source = ("prefix-" + secret + "-suffix").encode()

    first, first_truncated = redactor.redact_bytes_page(
        source,
        window_offset=0,
        page_offset=0,
        page_end=12,
        max_bytes=12,
        source_complete=True,
    )
    second, second_truncated = redactor.redact_bytes_page(
        source,
        window_offset=0,
        page_offset=12,
        page_end=len(source),
        max_bytes=len(source),
        source_complete=True,
    )

    assert secret.encode() not in first + second
    assert secret[:5].encode() not in first + second
    assert first_truncated is True
    assert second_truncated is True


def test_custom_truncation_marker_and_redaction_marker_remain_atomic() -> None:
    redactor = SecretRedactor("secret-value")
    value = "prefix-secret-value-suffix"

    projected, truncated = redactor.redact_text_bounded_with_marker(
        value,
        max_bytes=len("prefix-[REDA"),
        truncation_marker="[tool truncated]",
    )

    assert truncated is True
    assert "[REDA" not in projected
    assert "[tool trunc" not in projected
    assert len(projected.encode()) <= len("prefix-[REDA")
