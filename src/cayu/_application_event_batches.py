"""Which events ``CayuApp.emit_events`` and internal workflow emitters may append."""

from __future__ import annotations

from cayu.events import Event, validate_public_custom_event_type


def validate_workflow_event_batch(events: list[Event], *, allow_cayu_internal: bool) -> None:
    """Refuse anything but ``workflow.`` and ``custom.`` events.

    Application callers (``allow_cayu_internal=False``) must also use a public
    ``custom.`` name; Cayu's own emitters may use internal ones.
    """

    if type(events) is not list:
        raise TypeError("Runtime events must be a list.")
    for event in events:
        if type(event) is not Event:
            raise TypeError("Runtime events must be Event instances.")
        event_type = str(event.type)
        if not event_type.startswith(("workflow.", "custom.")):
            raise ValueError(
                "emit_events only accepts workflow. or custom. namespace "
                f"events; got {event_type!r}."
            )
        if not allow_cayu_internal and event_type.startswith("custom."):
            validate_public_custom_event_type(event_type)
