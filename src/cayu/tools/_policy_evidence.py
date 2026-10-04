from enum import StrEnum


class ToolPolicyEvidence(StrEnum):
    """Durable authority state for one policy-planned tool call.

    ``AUTHORITATIVE`` is the only state that may carry an executable policy
    decision. ``AMBIGUOUS`` means evaluation may have happened but its result
    was not durably published; operator acknowledgement cannot turn that
    missing evidence into authorization.
    """

    UNPLANNED = "unplanned"
    AUTHORITATIVE = "authoritative"
    UNREGISTERED = "unregistered"
    UNEXPOSED = "unexposed"
    AMBIGUOUS = "ambiguous"
