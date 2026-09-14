import pytest

from cayu.artifacts import (
    ArtifactClosureClaim,
    ArtifactClosureItem,
    copy_artifact_closure_claim,
)


def test_artifact_closure_contract_is_public():
    import cayu
    import cayu.artifacts as artifacts

    for name in ("ArtifactClosureClaim", "ArtifactClosureItem", "copy_artifact_closure_claim"):
        assert name in cayu.__all__
        assert name in artifacts.__all__
        assert getattr(cayu, name) is getattr(artifacts, name)


def test_artifact_closure_claim_detaches_exact_items():
    item = ArtifactClosureItem("artifact", 4, "a" * 64)
    claim = ArtifactClosureClaim("store", "session", "b" * 64, (item,))
    copied = copy_artifact_closure_claim(claim)
    assert copied == claim and copied is not claim
    assert copied.artifacts[0] is not item
    object.__setattr__(item, "size_bytes", 5)
    assert copied.artifacts[0].size_bytes == 4
    assert copy_artifact_closure_claim(claim) != copied


@pytest.mark.parametrize("sizes", [(True,), (-1,), (2**63,), (2**63 - 1, 1)])
def test_artifact_closure_claim_rejects_invalid_and_combined_sizes(sizes):
    with pytest.raises(ValueError):
        ArtifactClosureClaim(
            "store",
            "session",
            "b" * 64,
            tuple(
                ArtifactClosureItem(f"artifact-{index}", size, "a" * 64)
                for index, size in enumerate(sizes)
            ),
        )


@pytest.mark.parametrize("ids", [("same", "same"), ("z", "a")])
def test_artifact_closure_claim_requires_unique_sorted_ids(ids):
    with pytest.raises(ValueError, match="sorted and unique"):
        ArtifactClosureClaim(
            "store",
            "session",
            "b" * 64,
            tuple(ArtifactClosureItem(identity, 1, "a" * 64) for identity in ids),
        )


@pytest.mark.parametrize("field", ["artifact_id", "size_bytes", "metadata_sha256"])
def test_artifact_closure_claim_rejects_mutation_without_diagnostic_payload(
    field, caplog, capsys, recwarn
):
    class Private:
        def __repr__(self):
            return "private-artifact-closure-canary"

    item = ArtifactClosureItem("artifact", 4, "a" * 64)
    claim = ArtifactClosureClaim("store", "session", "b" * 64, (item,))
    object.__setattr__(item, field, Private())
    with pytest.raises(ValueError) as error:
        copy_artifact_closure_claim(claim)
    captured = capsys.readouterr()
    assert "private-artifact-closure-canary" not in (
        str(error.value)
        + captured.out
        + captured.err
        + caplog.text
        + "".join(str(item.message) for item in recwarn)
    )
