"""Bounded history pins derived only from typed committed owner receipts."""

from typing import Literal

from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._permits import PermitReceipt
from cayu.collaboration.lifecycle import LifecycleReceipt
from cayu.collaboration.participants import ParticipantReceipt

HistoryFamily = Literal["configurations", "lifecycle_history"]
HistoryKey = tuple[HistoryFamily, str, int]


def history_references(value: ContractValue) -> tuple[HistoryKey, ...]:
    snapshots = ()
    if isinstance(value, ParticipantReceipt):
        snapshots = value.participants
    elif isinstance(value, LifecycleReceipt) and value.participant is not None:
        snapshots = (value.participant,)
    refs: set[HistoryKey] = set()
    for snapshot in snapshots:
        refs.add(
            ("configurations", snapshot.reference.participant_id, snapshot.configuration_revision)
        )
        refs.add(
            ("lifecycle_history", snapshot.reference.participant_id, snapshot.lifecycle_revision)
        )
    if isinstance(value, PermitReceipt):
        request = value.expected.intent.request
        refs.add(
            (
                "lifecycle_history",
                request.participant.participant_id,
                request.expected_lifecycle_revision,
            )
        )
    return tuple(sorted(refs))
