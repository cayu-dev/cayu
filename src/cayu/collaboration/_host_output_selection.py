"""Select one protocol step from authenticated, content-free source evidence.

Selection creates no permission. The selected existing receiving owner repeats
its current authorization and exact-operation checks when dispatched.
"""

from dataclasses import dataclass
from typing import Literal

from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._producer_inspection import ProducerOutputInspection


@dataclass(frozen=True, slots=True)
class HostOutputSelection:
    intent: HostProducerMaintenance | None
    blocked: Literal["completion", "closure", "destination_cleanup", "destination", "state"] | None


def select_producer_output(snapshot: ProducerOutputInspection) -> HostOutputSelection:
    """Pure protocol selection; domain routing and disclosure remain external."""
    if snapshot.cleanup_ack is not None:
        return HostOutputSelection(None, None)

    def choose(action, destination=None):
        return HostOutputSelection(
            HostProducerMaintenance(
                recovery=snapshot.recovery, action=action, destination=destination
            ),
            None,
        )

    if snapshot.request_state in ("cancelled", "expired"):
        # Current closure/disposition and native exclusion/release need their
        # own positive reconciliation, not an invented successful answer.
        return HostOutputSelection(None, "closure")
    if snapshot.completion is None or snapshot.completion_disposition is None:
        if snapshot.state == "launch_claimed" and snapshot.completion is None:
            # Observe the original native owner. Missing output is pending, not
            # permission to redispatch, nor evidence of successful cleanup.
            return choose("retain_completion")
        return HostOutputSelection(None, "completion")
    if snapshot.request_state == "open":
        if snapshot.completion_disposition != "answer":
            return choose("publish_failure")
        for destination in snapshot.destinations:
            if destination.delivery == "excluded":
                continue
            if destination.export == "rejected":
                return choose("publish_failure")
            return choose(
                "publish_answer" if destination.export == "published" else "export",
                destination.destination,
            )
        return HostOutputSelection(None, "destination")
    if snapshot.request_state == "failed":
        if snapshot.completion_disposition != "answer":
            return choose("settle")
        for destination in snapshot.destinations:
            if destination.delivery != "excluded":
                return choose("exclude", destination.destination)
            if destination.export is not None and destination.export_cleanup is None:
                return choose("retire", destination.destination)
        return choose("settle")
    if snapshot.request_state != "answered":
        return HostOutputSelection(None, "state")
    for destination in snapshot.destinations:
        if destination.delivery == "appended":
            if destination.export_cleanup is None:
                return choose("release_export", destination.destination)
            continue
        if destination.delivery == "excluded":
            if destination.export is not None and destination.export_cleanup is None:
                return choose("retire", destination.destination)
            continue
        if destination.export == "rejected":
            return HostOutputSelection(None, "destination_cleanup")
        return choose(
            "deliver" if destination.export == "published" else "export",
            destination.destination,
        )
    return choose("settle")
