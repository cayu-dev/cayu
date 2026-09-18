"""Bounded history pins derived only from typed committed owner receipts."""

from typing import Literal

from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._permits import PermitReceipt
from cayu.collaboration.lifecycle import LifecycleReceipt
from cayu.collaboration.participants import ParticipantReceipt
from cayu.collaboration.requests import (
    RequestAdmissionReceipt,
    RequestControlReceipt,
    RequestObservationReceipt,
    RequestOutcomeReceipt,
    RequestProgressReceipt,
    RequestReceipt,
)

HistoryFamily = Literal["configurations", "lifecycle_history"]
HistoryKey = tuple[HistoryFamily, str, int]


def history_references(value: ContractValue) -> tuple[HistoryKey, ...]:
    snapshots = ()
    if isinstance(value, ParticipantReceipt):
        snapshots = value.participants
    elif isinstance(value, LifecycleReceipt) and value.participant is not None:
        snapshots = (value.participant,)
    elif isinstance(
        value,
        (
            RequestReceipt,
            RequestControlReceipt,
            RequestAdmissionReceipt,
            RequestProgressReceipt,
            RequestOutcomeReceipt,
            RequestObservationReceipt,
        ),
    ):
        command = (
            value.expected
            if isinstance(value, RequestReceipt)
            else value.expected.intent.expected
            if isinstance(value, RequestControlReceipt)
            else value.command.expected
            if isinstance(
                value, (RequestAdmissionReceipt, RequestProgressReceipt, RequestOutcomeReceipt)
            )
            else value.expected
        )
        selected = command.intent.selection
        snapshots = (selected.sender, selected.recipient)
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
