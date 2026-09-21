from __future__ import annotations

import pytest

from cayu.artifacts._input_manifest import FolderInputManifest
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.vaults.redaction import SecretRedactor


def entry(path="file", *, size=4, digest="a" * 64):
    return {
        "path": path,
        "git_mode": "100644",
        "member": {
            "resource": {
                "owner": {"application_scope": "app", "owner_id": "store", "incarnation": "one"},
                "kind": "artifact",
                "object_id": "art_" + "1" * 32,
                "incarnation": "original",
                "revision": 1,
            },
            "content_sha256": digest,
            "metadata_sha256": "b" * 64,
            "size_bytes": size,
        },
    }


def prepared(entries):
    return prepare_contract(FolderInputManifest, {"entries": entries}, redactor=SecretRedactor())


def test_manifest_roundtrip_and_distinct_retention_accounting():
    manifest = prepared([entry("a"), entry("b")])
    assert manifest.logical_bytes == 8
    assert len(manifest.retained_members) == 1
    assert sum(member.size_bytes for member in manifest.retained_members) == 4
    encoded = contract_bytes(manifest, redactor=SecretRedactor())
    assert FolderInputManifest.model_validate_json(encoded) == manifest
    assert prepared([]).retained_members == ()


@pytest.mark.parametrize("path", ["/a", "a/", "a//b", "a/../b", "./a", "a\\b", "C:a", "a\n"])
def test_reject_noncanonical_path(path):
    with pytest.raises(CollaborationContractError):
        prepared([entry(path)])


@pytest.mark.parametrize("paths", [["a", "a"], ["b", "a"], ["a", "a/b"]])
def test_reject_ambiguous_path_sets(paths):
    with pytest.raises(CollaborationContractError):
        prepared([entry(path) for path in paths])


@pytest.mark.parametrize(
    "entries",
    [
        [entry("a"), entry("b", digest="c" * 64)],
        [entry("a", digest="c" * 64), entry("b")],
        [entry(size=True)],
        [entry(size=-1)],
    ],
)
def test_reject_conflicting_or_invalid_members(entries):
    with pytest.raises(CollaborationContractError):
        prepared(entries)


@pytest.mark.parametrize("count", [31, 32, 33])
def test_member_count_bound(count):
    entries = [entry(f"f{index:02}") for index in range(count)]
    if count > 32:
        with pytest.raises(CollaborationContractError):
            prepared(entries)
    else:
        assert len(prepared(entries).entries) == count


def test_different_material_owners_are_not_implicitly_qualified():
    other = entry("b")
    other["member"]["resource"]["owner"]["incarnation"] = "replacement"
    with pytest.raises(CollaborationContractError):
        prepared([entry("a"), other])


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 2])
def test_schema_version_is_strict(version):
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            FolderInputManifest,
            {"schema_version": version, "entries": []},
            redactor=SecretRedactor(),
        )
