"""Resolving the Release a host states for one application process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.releases import (
    RELEASE_ID_ENV,
    RELEASE_ID_MAX_CHARS,
    RELEASE_SNAPSHOT_ROOT_ENV,
    ReleaseIdentity,
    recorded_release_identity,
    release_event_fields,
    resolve_release_identity,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import SupervisionEventQuery
from cayu.vaults.redaction import SecretRedactor

SNAPSHOT_ROOT = "5f0c" + "e1" * 30


class _VersionedProvider(ModelProvider):
    name = "versioned"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_a_process_without_a_stated_release_records_none() -> None:
    app = CayuApp(enable_logging=False)

    assert app.release_identity is None
    assert release_event_fields(app.release_identity) == {}


def test_the_environment_states_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RELEASE_ID_ENV, "dep_7c1d")
    monkeypatch.setenv(RELEASE_SNAPSHOT_ROOT_ENV, SNAPSHOT_ROOT)

    app = CayuApp(enable_logging=False)

    assert app.release_identity == ReleaseIdentity(id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT)


def test_a_release_id_alone_records_no_snapshot_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RELEASE_ID_ENV, "dep_7c1d")

    assert CayuApp(enable_logging=False).release_identity == ReleaseIdentity(id="dep_7c1d")


def test_configuration_states_the_release_without_the_environment() -> None:
    configured = ReleaseIdentity(id="release-configured", snapshot_root=SNAPSHOT_ROOT)

    assert CayuApp(enable_logging=False, release_identity=configured).release_identity == (
        configured
    )


def test_configuration_that_agrees_with_the_environment_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RELEASE_ID_ENV, "dep_7c1d")
    monkeypatch.setenv(RELEASE_SNAPSHOT_ROOT_ENV, SNAPSHOT_ROOT)
    stated = ReleaseIdentity(id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT)

    assert CayuApp(enable_logging=False, release_identity=stated).release_identity == stated


@pytest.mark.parametrize(
    "configured",
    [
        ReleaseIdentity(id="release-configured", snapshot_root=SNAPSHOT_ROOT),
        ReleaseIdentity(id="dep_7c1d", snapshot_root="0" * 64),
        ReleaseIdentity(id="dep_7c1d"),
    ],
    ids=["other-id", "other-root", "missing-root"],
)
def test_configuration_cannot_replace_the_environments_release(
    monkeypatch: pytest.MonkeyPatch,
    configured: ReleaseIdentity,
) -> None:
    monkeypatch.setenv(RELEASE_ID_ENV, "dep_7c1d")
    monkeypatch.setenv(RELEASE_SNAPSHOT_ROOT_ENV, SNAPSHOT_ROOT)

    names_both_variables = f"{RELEASE_ID_ENV} and {RELEASE_SNAPSHOT_ROOT_ENV}"

    with pytest.raises(ValueError, match=names_both_variables) as excinfo:
        CayuApp(enable_logging=False, release_identity=configured)

    for value in ("dep_7c1d", SNAPSHOT_ROOT, configured.id):
        assert value not in str(excinfo.value)


@pytest.mark.parametrize(
    "configured", [None, ReleaseIdentity(id="dep_7c1d")], ids=["unconfigured", "configured"]
)
def test_a_snapshot_root_without_a_release_id_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    configured: ReleaseIdentity | None,
) -> None:
    monkeypatch.setenv(RELEASE_SNAPSHOT_ROOT_ENV, SNAPSHOT_ROOT)

    with pytest.raises(ValueError, match=f"{RELEASE_SNAPSHOT_ROOT_ENV} requires {RELEASE_ID_ENV}"):
        CayuApp(enable_logging=False, release_identity=configured)


@pytest.mark.parametrize(
    ("name", "value", "release_id"),
    [
        (RELEASE_ID_ENV, "do-not-print\n", None),
        (RELEASE_ID_ENV, "x" * (RELEASE_ID_MAX_CHARS + 1), None),
        (RELEASE_ID_ENV, "", None),
        (RELEASE_ID_ENV, "do-not\nprint", None),
        (RELEASE_SNAPSHOT_ROOT_ENV, "DO-NOT-PRINT", "dep_7c1d"),
        (RELEASE_SNAPSHOT_ROOT_ENV, SNAPSHOT_ROOT.upper(), "dep_7c1d"),
    ],
    ids=[
        "untrimmed-id",
        "overlong-id",
        "blank-id",
        "control-character-id",
        "non-hex-root",
        "uppercase-root",
    ],
)
@pytest.mark.parametrize(
    "configured", [None, ReleaseIdentity(id="dep_7c1d")], ids=["unconfigured", "configured"]
)
def test_invalid_environment_values_are_rejected_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    release_id: str | None,
    configured: ReleaseIdentity | None,
) -> None:
    monkeypatch.setenv(name, value)
    if release_id is not None:
        monkeypatch.setenv(RELEASE_ID_ENV, release_id)

    with pytest.raises(ValueError, match=name) as excinfo:
        CayuApp(enable_logging=False, release_identity=configured)

    if value:
        assert value not in str(excinfo.value)


@pytest.mark.parametrize(
    "fields",
    [
        {"id": " padded "},
        {"id": "x" * (RELEASE_ID_MAX_CHARS + 1)},
        {"id": "dep\u00857c1d"},
        {"id": "dep_7c1d", "snapshot_root": "not-a-digest"},
        {"id": "dep_7c1d", "extra": "field"},
    ],
    ids=["untrimmed-id", "overlong-id", "control-character-id", "non-hex-root", "unknown-field"],
)
def test_configured_identity_is_validated(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        ReleaseIdentity.model_validate(fields)


@pytest.mark.parametrize(
    "fields",
    [{"id": " padded "}, {"id": "dep\u00857c1d"}, {"id": "dep_7c1d", "snapshot_root": "x"}],
    ids=["untrimmed-id", "control-character-id", "non-hex-root"],
)
def test_an_unvalidated_identity_is_validated_again(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        CayuApp(enable_logging=False, release_identity=ReleaseIdentity.model_construct(**fields))


def test_a_supervision_filter_rejects_an_invalid_release_id() -> None:
    window = {
        "start_at": datetime(2026, 1, 1, tzinfo=UTC),
        "end_at": datetime(2026, 1, 2, tzinfo=UTC),
    }

    assert SupervisionEventQuery(**window, release_id="dep_7c1d").release_id == "dep_7c1d"
    for release_id in (" padded ", "dep\n7c1d", "x" * (RELEASE_ID_MAX_CHARS + 1)):
        with pytest.raises(ValueError):
            SupervisionEventQuery(**window, release_id=release_id)


def test_the_environment_can_be_supplied_explicitly() -> None:
    stated = {RELEASE_ID_ENV: "dep_7c1d", RELEASE_SNAPSHOT_ROOT_ENV: SNAPSHOT_ROOT}

    assert resolve_release_identity(None, SecretRedactor(), environ=stated) == ReleaseIdentity(
        id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT
    )
    assert resolve_release_identity(None, SecretRedactor(), environ={}) is None


def test_a_release_naming_a_workload_secret_is_rejected() -> None:
    with pytest.raises(ValueError, match="contains a workload secret"):
        CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("secret-release-value"),
            release_identity=ReleaseIdentity(id="secret-release-value"),
        )


@pytest.mark.parametrize("secret", ["release", "id", "root"])
def test_a_secret_that_only_matches_a_recorded_field_name_is_accepted(secret: str) -> None:
    stated = ReleaseIdentity(id="dep_7c1d")

    assert resolve_release_identity(stated, SecretRedactor(secret), environ={}) == stated


@pytest.mark.parametrize(
    ("secret", "release"),
    [
        ("7c1d", ReleaseIdentity(id="dep_7c1d")),
        ("e1e1e1e1", ReleaseIdentity(id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT)),
    ],
    ids=["id", "snapshot-root"],
)
def test_a_secret_inside_a_release_value_is_rejected(secret: str, release: ReleaseIdentity) -> None:
    with pytest.raises(ValueError, match="contains a workload secret"):
        resolve_release_identity(release, SecretRedactor(secret), environ={})


def test_the_release_never_changes_the_execution_profile() -> None:
    async def candidate_fingerprint(release: ReleaseIdentity | None) -> str:
        app = CayuApp(enable_logging=False, release_identity=release)
        app.register_provider(_VersionedProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fixture-model"))
        candidate = await app.inspect_candidate_execution_profile(
            "assistant", causal_budget_id="budget-1"
        )
        return candidate.execution_profile.fingerprint

    fingerprints = {
        asyncio.run(candidate_fingerprint(release))
        for release in (
            None,
            ReleaseIdentity(id="release-1"),
            ReleaseIdentity(id="release-2", snapshot_root=SNAPSHOT_ROOT),
        )
    }

    assert len(fingerprints) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"release": None},
        {"release": "dep_7c1d"},
        {"release": {"id": " padded "}},
        {"release": {"id": "dep_7c1d", "snapshot_root": "not-a-digest"}},
    ],
    ids=["absent", "null", "text", "untrimmed-id", "non-hex-root"],
)
def test_a_start_event_without_a_valid_release_records_none(payload: dict) -> None:
    assert recorded_release_identity(payload) is None


def test_recorded_fields_round_trip() -> None:
    release = ReleaseIdentity(id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT)

    assert recorded_release_identity(release_event_fields(release)) == release


def test_a_release_recorded_with_fields_this_version_does_not_know_is_still_read() -> None:
    payload = {"release": {"id": "dep_7c1d", "snapshot_root": SNAPSHOT_ROOT, "channel": "canary"}}

    assert recorded_release_identity(payload) == ReleaseIdentity(
        id="dep_7c1d", snapshot_root=SNAPSHOT_ROOT
    )
