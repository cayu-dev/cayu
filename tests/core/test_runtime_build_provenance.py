from __future__ import annotations

import json
from base64 import urlsafe_b64encode
from hashlib import sha256

import pytest

import cayu.build_provenance as build_provenance_module
from cayu.agent_snapshots import execution_profile_snapshot_ref
from cayu.build_provenance import (
    RUNTIME_BUILD_PROVENANCE_ENV,
    RUNTIME_BUILD_PROVENANCE_STRICT_ENV,
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceAvailability,
    RuntimeBuildProvenanceOrigin,
    RuntimeBuildProvenanceStrength,
    _development_source_tree_provenance,
    _wheel_record_provenance,
    current_runtime_build_provenance,
    legacy_runtime_build_provenance,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentityAvailability,
    build_execution_profile_identity,
    changed_execution_profile_components,
)
from cayu.runtime.sessions import runtime_build_provenance_from_session_metadata


def _artifact_provenance(
    marker: str,
    *,
    source_revision: str | None = None,
    origin: RuntimeBuildProvenanceOrigin = RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
    artifact_kind: RuntimeBuildArtifactKind = RuntimeBuildArtifactKind.OTHER,
) -> RuntimeBuildProvenance:
    return RuntimeBuildProvenance.from_artifact_digest(
        origin=origin,
        artifact_kind=artifact_kind,
        artifact_digest=sha256(marker.encode()).hexdigest(),
        source_revision=source_revision,
    )


def _profile(provenance: RuntimeBuildProvenance):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="0.4.0",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
        runtime_build_provenance=provenance,
    )


def test_same_runtime_version_with_different_builds_changes_runtime_profile() -> None:
    build_a = _profile(_artifact_provenance("build-a"))
    build_b = _profile(_artifact_provenance("build-b"))

    assert build_a.fingerprint != build_b.fingerprint
    assert changed_execution_profile_components(build_a, build_b) == (
        ExecutionProfileComponentClass.RUNTIME,
    )


def test_source_revision_is_diagnostic_not_runtime_identity() -> None:
    first = _artifact_provenance("same-artifact", source_revision="revision-a")
    second = _artifact_provenance("same-artifact", source_revision="revision-b")

    assert first.fingerprint == second.fingerprint
    assert _profile(first) == _profile(second)


def test_snapshot_profile_retains_reconstructible_build_provenance() -> None:
    provenance = _artifact_provenance("portable-build", source_revision="revision-a")
    profile = _profile(provenance)
    snapshot_profile = execution_profile_snapshot_ref(profile)

    assert profile.runtime_build_provenance == provenance.model_copy(
        update={"source_revision": None}
    )
    assert snapshot_profile.runtime_build_provenance == profile.runtime_build_provenance


def test_missing_build_provenance_is_unavailable_not_a_wildcard() -> None:
    legacy = runtime_build_provenance_from_session_metadata({})
    profile = _profile(legacy)
    runtime_component = profile.component(ExecutionProfileComponentClass.RUNTIME)

    assert legacy.origin is RuntimeBuildProvenanceOrigin.LEGACY_RECORD
    assert legacy.availability is RuntimeBuildProvenanceAvailability.UNAVAILABLE
    assert runtime_component.availability is ExecutionProfileIdentityAvailability.UNAVAILABLE
    assert legacy == legacy_runtime_build_provenance()


def test_wheel_record_projection_covers_code_and_distribution_metadata() -> None:
    class WheelHash:
        mode = "sha256"

        def __init__(self, value: str) -> None:
            self.value = value

    class WheelFile:
        def __init__(self, path: str, value: str | None, size: int) -> None:
            self.path = path
            self.hash = None if value is None else WheelHash(value)
            self.size = size

        def __str__(self) -> str:
            return self.path

    class Distribution:
        def __init__(self, metadata_hash: str) -> None:
            self.files = (
                WheelFile("cayu/runtime/app.py", "code-hash", 123),
                WheelFile(
                    "cayu/runtime/__pycache__/app.cpython-314.pyc",
                    None,
                    234,
                ),
                WheelFile("cayu-0.4.0.dist-info/METADATA", metadata_hash, 45),
                WheelFile("cayu-0.4.0.dist-info/RECORD", None, 67),
            )

    first = _wheel_record_provenance(
        Distribution("metadata-a")  # ty: ignore[invalid-argument-type]
    )
    changed = _wheel_record_provenance(
        Distribution("metadata-b")  # ty: ignore[invalid-argument-type]
    )

    assert first is not None
    assert changed is not None
    assert first.origin is RuntimeBuildProvenanceOrigin.WHEEL_RECORD
    assert first.strength is RuntimeBuildProvenanceStrength.STRUCTURAL
    assert first.fingerprint != changed.fingerprint


def test_wheel_record_projection_covers_hashed_runtime_bytecode() -> None:
    class WheelHash:
        mode = "sha256"

        def __init__(self, value: str) -> None:
            self.value = value

    class WheelFile:
        def __init__(self, path: str, value: str | None, size: int) -> None:
            self.path = path
            self.hash = None if value is None else WheelHash(value)
            self.size = size

        def __str__(self) -> str:
            return self.path

    class Distribution:
        def __init__(self, bytecode_hash: str) -> None:
            self.files = (
                WheelFile(
                    "cayu/runtime/__pycache__/app.cpython-314.pyc",
                    bytecode_hash,
                    234,
                ),
                WheelFile("cayu-0.4.0.dist-info/METADATA", "metadata", 45),
                WheelFile("cayu-0.4.0.dist-info/RECORD", None, 67),
            )

    first = _wheel_record_provenance(
        Distribution("bytecode-a")  # ty: ignore[invalid-argument-type]
    )
    changed = _wheel_record_provenance(
        Distribution("bytecode-b")  # ty: ignore[invalid-argument-type]
    )

    assert first is not None
    assert changed is not None
    assert first.fingerprint != changed.fingerprint


def test_wheel_record_projection_rejects_uncovered_unhashed_bytecode() -> None:
    class WheelFile:
        hash = None
        size = 234

        def __str__(self) -> str:
            return "cayu/runtime/__pycache__/app.cpython-314.pyc"

    class Distribution:
        files = (WheelFile(),)

    assert (
        _wheel_record_provenance(
            Distribution()  # ty: ignore[invalid-argument-type]
        )
        is None
    )


@pytest.mark.parametrize("uncovered_hash", [None, "unsupported"])
def test_wheel_record_projection_rejects_uncovered_behavior_files(
    monkeypatch,
    uncovered_hash: str | None,
) -> None:
    class WheelHash:
        def __init__(self, mode: str, value: str) -> None:
            self.mode = mode
            self.value = value

    class WheelFile:
        def __init__(self, path: str, file_hash: WheelHash | None, size: int) -> None:
            self.path = path
            self.hash = file_hash
            self.size = size

        def __str__(self) -> str:
            return self.path

    class Distribution:
        files = (
            WheelFile("cayu/__init__.py", WheelHash("sha256", "covered"), 10),
            WheelFile(
                "cayu/runtime/app.py",
                (None if uncovered_hash is None else WheelHash("sha512", "not-authoritative")),
                20,
            ),
            WheelFile("cayu-0.4.0.dist-info/RECORD", None, 30),
        )

    distribution = Distribution()
    assert (
        _wheel_record_provenance(
            distribution  # ty: ignore[invalid-argument-type]
        )
        is None
    )

    monkeypatch.setattr(
        build_provenance_module.metadata, "distribution", lambda _name: distribution
    )
    monkeypatch.setattr(build_provenance_module, "_distribution_is_editable", lambda _value: False)
    monkeypatch.setattr(
        build_provenance_module,
        "_distribution_owns_runtime_module",
        lambda _distribution, _runtime_module: True,
    )
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_STRICT_ENV, "1")
    monkeypatch.delenv(RUNTIME_BUILD_PROVENANCE_ENV, raising=False)
    current_runtime_build_provenance.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="only available/application_versioned evidence"):
            current_runtime_build_provenance()
    finally:
        current_runtime_build_provenance.cache_clear()


def test_source_checkout_ignores_an_unrelated_same_name_wheel_distribution(
    tmp_path,
    monkeypatch,
) -> None:
    unrelated_site = tmp_path / "unrelated-site"
    unrelated_package = unrelated_site / "cayu"
    unrelated_distribution = unrelated_site / "cayu-9.9.9.dist-info"
    unrelated_package.mkdir(parents=True)
    unrelated_distribution.mkdir()
    fake_module = unrelated_package / "build_provenance.py"
    fake_module.write_text("UNRELATED = True\n", encoding="utf-8")
    metadata_file = unrelated_distribution / "METADATA"
    metadata_file.write_text(
        "Metadata-Version: 2.1\nName: cayu\nVersion: 9.9.9\n",
        encoding="utf-8",
    )

    def record_entry(path, content: bytes) -> str:
        digest = urlsafe_b64encode(sha256(content).digest()).rstrip(b"=").decode("ascii")
        return f"{path},sha256={digest},{len(content)}"

    (unrelated_distribution / "RECORD").write_text(
        "\n".join(
            (
                record_entry("cayu/build_provenance.py", fake_module.read_bytes()),
                record_entry("cayu-9.9.9.dist-info/METADATA", metadata_file.read_bytes()),
                "cayu-9.9.9.dist-info/RECORD,,",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(unrelated_site))
    monkeypatch.delenv(RUNTIME_BUILD_PROVENANCE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_BUILD_PROVENANCE_STRICT_ENV, raising=False)
    current_runtime_build_provenance.cache_clear()
    try:
        observed = current_runtime_build_provenance()
    finally:
        current_runtime_build_provenance.cache_clear()

    assert observed.origin is RuntimeBuildProvenanceOrigin.DEVELOPMENT_SOURCE_TREE
    assert observed.artifact_kind is RuntimeBuildArtifactKind.SOURCE_TREE


def test_development_tree_identity_is_deterministic_and_explicitly_weak(tmp_path) -> None:
    package_root = tmp_path / "cayu"
    package_root.mkdir()
    (package_root / "runtime.py").write_text("BUILD = 'a'\n", encoding="utf-8")

    first = _development_source_tree_provenance(package_root)
    repeated = _development_source_tree_provenance(package_root)
    (package_root / "runtime.py").write_text("BUILD = 'b'\n", encoding="utf-8")
    changed = _development_source_tree_provenance(package_root)

    assert first == repeated
    assert first.origin is RuntimeBuildProvenanceOrigin.DEVELOPMENT_SOURCE_TREE
    assert first.strength is RuntimeBuildProvenanceStrength.APPLICATION_VERSIONED
    assert first.fingerprint != changed.fingerprint


def test_explicit_oci_manifest_is_loaded_once_and_preserves_digest_domain(
    monkeypatch,
) -> None:
    provenance = _artifact_provenance(
        "sha256:deployed-image",
        source_revision="0123456789abcdef",
        origin=RuntimeBuildProvenanceOrigin.OCI_IMAGE_DIGEST,
        artifact_kind=RuntimeBuildArtifactKind.OCI_IMAGE,
    )
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, provenance.model_dump_json())
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_STRICT_ENV, "1")
    current_runtime_build_provenance.cache_clear()
    try:
        observed = current_runtime_build_provenance()
        monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, json.dumps({"invalid": True}))
        repeated = current_runtime_build_provenance()
    finally:
        current_runtime_build_provenance.cache_clear()

    assert observed == provenance
    assert repeated == provenance
    assert observed.artifact_kind is RuntimeBuildArtifactKind.OCI_IMAGE


def test_strict_startup_rejects_weak_build_provenance(monkeypatch) -> None:
    weak = RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
        artifact_kind=RuntimeBuildArtifactKind.OTHER,
        artifact_digest=sha256(b"weak-explicit-build").hexdigest(),
        strength=RuntimeBuildProvenanceStrength.APPLICATION_VERSIONED,
    )
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, weak.model_dump_json())
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_STRICT_ENV, "1")
    current_runtime_build_provenance.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="only available/application_versioned evidence"):
            current_runtime_build_provenance()
    finally:
        current_runtime_build_provenance.cache_clear()
