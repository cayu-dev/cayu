"""Settlement shapes cannot infer receiving exclusion from missing delivery proof.

These are contract characterization checks; runtime retirement qualification is
in test_producer_export_retirement.py.
"""

import pytest

from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._producer_contracts import ProducerDestinationSettlement


def evidence():
    destination = OperationRef(
        application_scope="scope",
        namespace_incarnation="one",
        generation=1,
        caller_key="destination",
    )
    return dict(
        destination=destination,
        delivery=destination.model_copy(update={"caller_key": "delivery"}),
        receipt_commitment="sha256:" + "1" * 64,
        export_settlement_commitment="sha256:" + "2" * 64,
        export_state="released",
    )


def test_delivery_requires_positive_receiving_evidence():
    assert ProducerDestinationSettlement(**evidence()).kind == "delivery"
    for field in ("delivery", "receipt_commitment"):
        with pytest.raises(ValueError):
            ProducerDestinationSettlement(**(evidence() | {field: None}))


@pytest.mark.parametrize("state", ["excluded", "retired", "released"])
def test_retirement_has_explicit_native_state_and_no_delivery_receipt(state):
    fields = evidence() | dict(
        kind="export_retirement",
        delivery=None,
        receipt_commitment=None,
        export_state=state,
    )
    parsed = ProducerDestinationSettlement(**fields)
    assert ProducerDestinationSettlement.model_validate_json(parsed.model_dump_json()) == parsed
    for field in ("delivery", "receipt_commitment"):
        with pytest.raises(ValueError):
            ProducerDestinationSettlement(**(fields | {field: evidence()[field]}))


@pytest.mark.parametrize("state", [None, "pending", "prepared", "future", True])
def test_retirement_refuses_missing_or_nonterminal_native_state(state):
    with pytest.raises(ValueError):
        ProducerDestinationSettlement(
            **(
                evidence()
                | dict(
                    kind="export_retirement",
                    delivery=None,
                    receipt_commitment=None,
                    export_state=state,
                )
            )
        )


@pytest.mark.parametrize("state", [None, "excluded", "pending", "future", True])
def test_delivery_requires_terminal_published_export_state(state):
    with pytest.raises(ValueError):
        ProducerDestinationSettlement(**(evidence() | {"export_state": state}))


def test_delivery_retains_retired_export_state():
    assert (
        ProducerDestinationSettlement(**(evidence() | {"export_state": "retired"})).kind
        == "delivery"
    )
