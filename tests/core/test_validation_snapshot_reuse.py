from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BeforeValidator, StrictStr, ValidationError, model_validator

from cayu.collaboration import _contracts
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, ContractValue
from cayu.collaboration._validation_snapshot import current_validation_snapshots


class Leaf(ContractValue):
    text: StrictStr


class Branch(ContractValue):
    leaves: tuple[Leaf, ...]


class Tree(ContractValue):
    branches: tuple[Branch, ...]


def test_nested_validation_admits_json_once_but_still_builds_detached_records(monkeypatch):
    calls = []
    inspect = _contracts.inspect_bounded_durable_json

    def observe(value, *args, **kwargs):
        calls.append(value)
        return inspect(value, *args, **kwargs)

    monkeypatch.setattr(_contracts, "inspect_bounded_durable_json", observe)
    raw = {"branches": [{"leaves": [{"text": "one"}, {"text": "two"}]}]}
    tree = Tree.model_validate(raw)
    assert len(calls) == 1
    raw["branches"][0]["leaves"][0]["text"] = "changed"
    assert tree.branches[0].leaves[0].text == "one"
    assert Tree.model_validate(raw).branches[0].leaves[0].text == "changed"
    assert len(calls) == 2


@pytest.mark.parametrize("invalid", ["\x00", "\ud800", "x" * MAX_ENVELOPE_BYTES])
@pytest.mark.parametrize("kind", ["before", "wrap", "annotation"])
def test_owner_mutation_cannot_reuse_parent_admission(kind, invalid):
    def mutate(value):
        assert type(value) is dict
        value["leaves"][0]["text"] = invalid
        return value

    if kind == "before":

        class MutatingBranch(Branch):
            @model_validator(mode="before")
            @classmethod
            def change(cls, value):
                return mutate(value)

        annotation = MutatingBranch
    elif kind == "wrap":

        class WrappingBranch(Branch):
            @model_validator(mode="wrap")
            @classmethod
            def change(cls, value, handler):
                handler(value)
                return handler(mutate(value))

        annotation = WrappingBranch
    else:
        annotation = Annotated[Branch, BeforeValidator(mutate)]

    class Owner(ContractValue):
        branch: annotation

    with pytest.raises(ValidationError):
        Owner.model_validate({"branch": {"leaves": [{"text": "safe"}]}})
    assert current_validation_snapshots() is None


def test_union_failure_cannot_bless_mutated_sibling_input():
    class MutatingBranch(Branch):
        @model_validator(mode="before")
        @classmethod
        def change(cls, value):
            value["leaves"][0]["text"] = "\x00"
            raise ValueError("try the next union branch")

    class Owner(ContractValue):
        branch: MutatingBranch | Branch

    with pytest.raises(ValidationError):
        Owner.model_validate({"branch": {"leaves": [{"text": "safe"}]}})


def test_owner_validators_run_again_for_each_identical_input():
    calls = []

    class Checked(Leaf):
        @model_validator(mode="after")
        def decide(self):
            calls.append(self.text)
            if len(calls) == 2:
                raise ValueError("owner decision changed")
            return self

    class Owner(ContractValue):
        leaves: tuple[Checked, ...]

    raw = {"text": "same"}
    with pytest.raises(ValidationError, match="owner decision changed"):
        Owner.model_validate({"leaves": [raw, raw]})
    assert calls == ["same", "same"]


def test_snapshot_capacity_falls_back_to_validation(monkeypatch):
    from cayu.collaboration import _validation_snapshot

    monkeypatch.setattr(_validation_snapshot, "_MAX_CONTAINERS", 1)
    assert Tree.model_validate({"branches": [{"leaves": [{"text": "safe"}]}]})
    with pytest.raises(ValidationError):
        Tree.model_validate({"branches": [{"leaves": [{"text": "\x00"}]}]})
    assert current_validation_snapshots() is None


def test_validation_scope_releases_snapshots_after_owner_failure():
    retained = []

    class Failing(Leaf):
        @model_validator(mode="after")
        def fail(self):
            retained.append(current_validation_snapshots())
            raise ValueError("failed")

    class Owner(ContractValue):
        child: Failing

    with pytest.raises(ValidationError):
        Owner.model_validate({"child": {"text": "private input"}})
    assert retained[0] is not None
    assert not retained[0].active
    assert not retained[0]._records
    assert current_validation_snapshots() is None
